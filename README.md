# robotic603_finalproject

robotic603_finalproject


In order to mount our `catkin_ws` directory within the docker container, we need to move it inside the newly cloned `stingray_docker` directory (due to docker permissions). Note that if we move `catkin_ws` as shown in the next line, we may need to run `catkin_make clean` (or `catkin clean`) before running `catkin_make` (or `catkin build`) again due to changes in the path.

```bash
mv ~/catkin_ws ~/stingray_docker/catkin_ws
catkin clean
catkin build
```
