#!/usr/bin/env python3
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from torch import nn


DEFAULT_CLASS_NAMES = ("circle", "square", "star", "triangle")
CNN_MEAN = (0.5, 0.5, 0.5)
CNN_STD = (0.5, 0.5, 0.5)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SymbolCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.30):
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 32),
            nn.MaxPool2d(kernel_size=2),
            self._conv_block(32, 64),
            nn.MaxPool2d(kernel_size=2),
            self._conv_block(64, 128),
            nn.MaxPool2d(kernel_size=2),
            self._conv_block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def torch_load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def infer_model_type(requested_type, checkpoint_path, checkpoint):
    if requested_type != "auto":
        return requested_type

    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    run_name = str(config.get("run_name", ""))
    path_text = str(checkpoint_path).lower()
    if "mobilenet" in path_text or "mobilenet" in run_name.lower():
        return "mobilenetv2"
    return "cnn"


def build_mobilenet_v2(num_classes, dropout):
    from torchvision.models import mobilenet_v2

    try:
        model = mobilenet_v2(weights=None)
    except TypeError:
        model = mobilenet_v2(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_model(model_type, num_classes, dropout):
    if model_type == "cnn":
        return SymbolCNN(num_classes=num_classes, dropout=dropout)
    if model_type == "mobilenetv2":
        return build_mobilenet_v2(num_classes=num_classes, dropout=dropout)
    raise ValueError("Unsupported model_type '{}'. Use cnn, mobilenetv2, or auto.".format(model_type))


def resolve_device(requested_device):
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ros_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def read_checkpoint_metadata(checkpoint, model_type, fallback_image_size):
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    image_size = int(config.get("image_size", fallback_image_size))
    dropout = float(config.get("dropout", 0.20 if model_type == "mobilenetv2" else 0.30))
    return list(class_names), image_size, dropout


def normalize_image(rgb_image, model_type, image_size):
    resized = cv2.resize(rgb_image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    array = resized.astype(np.float32) / 255.0
    if model_type == "mobilenetv2":
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
    else:
        mean = np.array(CNN_MEAN, dtype=np.float32)
        std = np.array(CNN_STD, dtype=np.float32)
    array = (array - mean) / std
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0)


def clamp(value, low, high):
    return max(low, min(high, value))


def detect_symbol_candidate(bgr_image, min_area_ratio, max_area_ratio, padding_ratio):
    height, width = bgr_image.shape[:2]
    image_area = float(width * height)
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, dark_mask = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(blurred, 70, 160)
    mask = cv2.bitwise_or(dark_mask, edges)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / image_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < 12 or box_height < 12:
            continue
        aspect_ratio = float(box_width) / float(box_height)
        if aspect_ratio < 0.25 or aspect_ratio > 4.0:
            continue
        center_x = x + box_width / 2.0
        center_y = y + box_height / 2.0
        center_penalty = abs(center_x - width / 2.0) / width
        score = area_ratio - 0.05 * center_penalty
        candidates.append((score, x, y, box_width, box_height, area_ratio))

    if not candidates:
        return None

    _, x, y, box_width, box_height, area_ratio = max(candidates, key=lambda item: item[0])
    padding = int(max(box_width, box_height) * padding_ratio)
    x1 = clamp(x - padding, 0, width - 1)
    y1 = clamp(y - padding, 0, height - 1)
    x2 = clamp(x + box_width + padding, x1 + 1, width)
    y2 = clamp(y + box_height + padding, y1 + 1, height)
    padded_width = x2 - x1
    padded_height = y2 - y1

    return {
        "x": int(x1),
        "y": int(y1),
        "width": int(padded_width),
        "height": int(padded_height),
        "center_x": float((x1 + padded_width / 2.0) / width),
        "center_y": float((y1 + padded_height / 2.0) / height),
        "area_ratio": float(area_ratio),
    }


class SymbolClassifierNode:
    def __init__(self):
        rospy.init_node("symbol_classifier")
        self.bridge = CvBridge()

        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.checkpoint_path = Path(
            rospy.get_param(
                "~checkpoint_path",
                os.environ.get("SYMBOL_MODEL_CHECKPOINT", "/nn_training/cnn/checkpoints/symbols_cnn_best.pt"),
            )
        ).expanduser()
        self.model_type = rospy.get_param("~model_type", "auto")
        self.device = resolve_device(rospy.get_param("~device", "auto"))
        self.fallback_image_size = int(rospy.get_param("~image_size", 128))
        self.max_rate = float(rospy.get_param("~max_rate", 5.0))
        self.min_confidence = float(rospy.get_param("~min_confidence", 0.0))
        self.publish_annotated = ros_bool(rospy.get_param("~publish_annotated", True))
        self.localize_symbol = ros_bool(rospy.get_param("~localize_symbol", True))
        self.min_candidate_area_ratio = float(rospy.get_param("~min_candidate_area_ratio", 0.002))
        self.max_candidate_area_ratio = float(rospy.get_param("~max_candidate_area_ratio", 0.65))
        self.candidate_padding_ratio = float(rospy.get_param("~candidate_padding_ratio", 0.18))
        self.last_inference_time = rospy.Time(0)

        self.label_pub = rospy.Publisher("~label", String, queue_size=10)
        self.confidence_pub = rospy.Publisher("~confidence", Float32, queue_size=10)
        self.result_pub = rospy.Publisher("~result", String, queue_size=10)
        self.annotated_pub = rospy.Publisher("~annotated_image", Image, queue_size=2)

        self.model, self.class_names, self.image_size = self.load_model()
        self.subscriber = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)

        rospy.loginfo(
            "symbol_classifier ready: topic=%s checkpoint=%s model=%s image_size=%d device=%s",
            self.image_topic,
            self.checkpoint_path,
            self.model_type,
            self.image_size,
            self.device,
        )

    def load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError("Model checkpoint does not exist: {}".format(self.checkpoint_path))

        checkpoint = torch_load_checkpoint(str(self.checkpoint_path), self.device)
        model_type = infer_model_type(self.model_type, self.checkpoint_path, checkpoint)
        class_names, image_size, dropout = read_checkpoint_metadata(checkpoint, model_type, self.fallback_image_size)
        model = build_model(model_type=model_type, num_classes=len(class_names), dropout=dropout)

        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.model_type = model_type
        return model, class_names, image_size

    def should_process(self):
        if self.max_rate <= 0:
            return True
        now = rospy.Time.now()
        if (now - self.last_inference_time).to_sec() < 1.0 / self.max_rate:
            return False
        self.last_inference_time = now
        return True

    def image_callback(self, msg):
        if not self.should_process():
            return

        try:
            bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(5.0, "Could not convert image: %s", exc)
            return

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        tensor = normalize_image(rgb_image, self.model_type, self.image_size).to(self.device)
        candidate = None
        if self.localize_symbol:
            candidate = detect_symbol_candidate(
                bgr_image,
                self.min_candidate_area_ratio,
                self.max_candidate_area_ratio,
                self.candidate_padding_ratio,
            )

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        predicted_label = self.class_names[best_index]
        published_label = predicted_label if confidence >= self.min_confidence else "unknown"

        result = {
            "label": published_label,
            "raw_label": predicted_label,
            "confidence": confidence,
            "accepted": bool(confidence >= self.min_confidence),
            "model_type": self.model_type,
            "checkpoint": str(self.checkpoint_path),
            "image_width": int(bgr_image.shape[1]),
            "image_height": int(bgr_image.shape[0]),
            "center_x": float(candidate["center_x"] if candidate else 0.5),
            "center_y": float(candidate["center_y"] if candidate else 0.5),
            "bbox": candidate,
            "stamp": msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec(),
            "probabilities": {
                class_name: float(probability)
                for class_name, probability in zip(self.class_names, probabilities)
            },
        }

        self.label_pub.publish(String(data=published_label))
        self.confidence_pub.publish(Float32(data=confidence))
        self.result_pub.publish(String(data=json.dumps(result, sort_keys=True)))
        rospy.loginfo_throttle(1.0, "symbol=%s confidence=%.3f", published_label, confidence)

        if self.publish_annotated and self.annotated_pub.get_num_connections() > 0:
            self.publish_annotated_image(msg, bgr_image, published_label, confidence, candidate)

    def publish_annotated_image(self, original_msg, bgr_image, label, confidence, candidate):
        annotated = bgr_image.copy()
        text = "{} {:.1f}%".format(label, confidence * 100.0)
        cv2.rectangle(annotated, (8, 8), (360, 54), (0, 0, 0), thickness=-1)
        cv2.putText(annotated, text, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        if candidate:
            x = int(candidate["x"])
            y = int(candidate["y"])
            width = int(candidate["width"])
            height = int(candidate["height"])
            cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 255), 2)
            center = (int(candidate["center_x"] * bgr_image.shape[1]), int(candidate["center_y"] * bgr_image.shape[0]))
            cv2.circle(annotated, center, 6, (0, 255, 255), -1)
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(5.0, "Could not publish annotated image: %s", exc)
            return
        annotated_msg.header = original_msg.header
        self.annotated_pub.publish(annotated_msg)


def main():
    try:
        SymbolClassifierNode()
        rospy.spin()
    except Exception as exc:
        rospy.logfatal("symbol_classifier failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
