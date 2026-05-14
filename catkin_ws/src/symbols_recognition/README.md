# symbols_recognition

ROS Noetic package for running symbol classification on the real Triton robot.

## Docker

From the repository root:

```bash
docker compose build
ROS_IP=<robot-or-host-ip> docker compose up -d
docker compose exec triton_noetic_ping /setup.bash
```

`docker-compose.yaml` mounts `./nn_training` at `/nn_training` so trained checkpoints are available to the ROS node.

To open a shell instead:

```bash
docker compose exec triton_noetic_ping bash
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash
roslaunch symbols_recognition real_robot.launch
```

## Launch

The real robot entrypoint is:

```bash
roslaunch symbols_recognition real_robot.launch
```

It starts the Triton hardware bringup from `stingray_camera`, subscribes to the RealSense color image, loads a trained PyTorch checkpoint, and publishes symbol predictions.

Default topics:

- input image: `/camera/color/image_raw`
- prediction label: `/symbol_classifier/label`
- confidence: `/symbol_classifier/confidence`
- full JSON result: `/symbol_classifier/result`
- annotated image: `/symbol_classifier/annotated_image`

Default checkpoint inside Docker:

```text
/nn_training/cnn/checkpoints/symbols_cnn_best.pt
```

Use MobileNetV2 instead:

```bash
roslaunch symbols_recognition real_robot.launch \
  model_type:=mobilenetv2 \
  checkpoint_path:=/nn_training/mobilenetv2/checkpoints/symbols_mobilenetv2_best.pt
```

Run only the detector when the Triton hardware stack is already running:

```bash
roslaunch symbols_recognition real_robot.launch bringup:=false
```

Enable the optional real-robot LiDAR wall follower:

```bash
roslaunch symbols_recognition real_robot.launch wall_follow:=true
```
