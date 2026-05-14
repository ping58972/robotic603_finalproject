# symbols_recognition

ROS Noetic package for running symbol classification on the real Triton robot.

## Docker

From the repository root:

```bash
docker compose build
ROS_IP=<robot-or-host-ip> docker compose up -d
docker compose exec triton_noetic_ping /setup.bash
```

`/setup.bash` prompts for the target symbol:

```text
Target symbol [star, square, circle, triangle]:
```

You can also pass it directly:

```bash
docker compose exec triton_noetic_ping /setup.bash target_symbol:=star
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

It starts the Triton hardware bringup from `stingray_camera`, subscribes to the RealSense color image, loads a trained PyTorch checkpoint, publishes symbol predictions, and starts the target-seeking controller.

The target-seeking behavior is:

1. Read a target symbol: `star`, `square`, `circle`, or `triangle`.
2. Rotate in place until the classifier sees the target symbol.
3. Align the target toward the camera center.
4. Drive forward while the target stays visible.
5. Stop when the front LiDAR range is within `stop_distance`.

Direct launch with a target symbol:

```bash
roslaunch symbols_recognition real_robot.launch target_symbol:=square
```

Disable target seeking and only publish classifier results:

```bash
roslaunch symbols_recognition real_robot.launch target_seek:=false
```

Default topics:

- input image: `/camera/color/image_raw`
- prediction label: `/symbol_classifier/label`
- confidence: `/symbol_classifier/confidence`
- full JSON result: `/symbol_classifier/result`
- annotated image: `/symbol_classifier/annotated_image`
- robot velocity command: `/cmd_vel`

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
roslaunch symbols_recognition real_robot.launch target_seek:=false wall_follow:=true
```

Useful target-seeking tuning args:

```bash
roslaunch symbols_recognition real_robot.launch \
  target_symbol:=circle \
  target_min_confidence:=0.80 \
  require_bbox:=true \
  require_classifier_ready:=true \
  classifier_timeout:=2.0 \
  stop_distance:=0.60 \
  approach_speed:=0.14 \
  search_angular_speed:=0.30
```

If RealSense prints messages such as `set_xu(ctrl=1) failed` or `requested device is NOT found`, the camera driver is not producing images. With `require_classifier_ready:=true`, the robot holds zero velocity until `/symbol_classifier/result` is live again. Check USB connection, camera permissions, and that another process is not already using the RealSense device.
