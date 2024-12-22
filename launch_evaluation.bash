#!/bin/bash

# Pass number of rollouts as argument
if [ $1 ]
then
  N="$1"
else
  N=5
fi

echo $2

if [ "$2" = "vision" ]
then
  echo
  echo "[LAUNCH SCRIPT] Vision based!"
  echo
  run_competition_args="--vision_based"
  realtimefactor="real_time_factor:=0.5"
elif [ "$2" = "state" ]
then
  echo
  echo "[LAUNCH SCRIPT] State based!"
  echo
  if [ "$3" = "human" ]
  then
    run_competition_args="--keyboard"
    realtimefactor=""
  else
    if [ "$3" = "planner" ]
    then
      run_competition_args="--planner"
      realtimefactor="real_time_factor:=1.0"
    else
      run_competition_args=""
      realtimefactor="real_time_factor:=1.0"
    fi
  fi
else
  echo
  echo "[LAUNCH SCRIPT] Unknown or empty second argument: $2, only 'vision' or 'state' allowed!"
  echo
  exit 1
fi

# Set Flightmare Path if it is not set
if [ -z $FLIGHTMARE_PATH ]
then
  export FLIGHTMARE_PATH=$PWD/flightmare
fi

# Launch the simulator, unless it is already running
if [ -z $(pgrep visionsim_node) ]
then
  roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=True $realtimefactor &
  ROS_PID="$!"
  echo $ROS_PID
  sleep 10
else
  ROS_PID=""
fi

SUMMARY_FILE="evaluation.yaml"
echo "" > $SUMMARY_FILE

# generate datetime string to label summary folders with in evaluation_node.py
datetime=$(date '+d%m_%d_t%H_%M')

relaunch_sim=0

# Perform N evaluation runs
for i in $(eval echo {0..$(($N-1))})
do

  # Reset the simulator if needed
  if ((relaunch_sim))
  then
      echo
      echo
      echo
      echo
      echo RELAUNCHING SIMULATOR ON RUN $i
      echo
      echo
      echo
      echo

    # reset flag and kill everything to restart
    relaunch_sim=0
    killall -9 roscore rosmaster rosout gzserver gzclient RPG_Flightmare.
    sleep 10

    # Launch the simulator, unless it is already running
    if [ -z $(pgrep visionsim_node) ]
    then
      roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=True $realtimefactor &
      ROS_PID="$!"
      echo $ROS_PID
      sleep 10
    else
      killall -9 roscore rosmaster rosout gzserver gzclient RPG_Flightmare.
      sleep 10
    fi

  fi

  start_time=$(date +%s)

  # Publish simulator reset
  rostopic pub /kingfisher/dodgeros_pilot/off std_msgs/Empty "{}" --once
  rostopic pub /kingfisher/dodgeros_pilot/reset_sim std_msgs/Empty "{}" --once
  rostopic pub /kingfisher/dodgeros_pilot/enable std_msgs/Bool "data: true" --once
  rostopic pub /kingfisher/dodgeros_pilot/start std_msgs/Empty "{}" --once

  export ROLLOUT_NAME="rollout_$(printf "%03d" "$i")"
  echo
  echo
  echo "$ROLLOUT_NAME"
  echo
  echo

  cd ./envtest/ros/
  python3 evaluation_node.py ${datetime}_N$(printf "%03d" "$i") &
  PY_PID="$!"

  python3 run_competition.py $run_competition_args --exp_name ${datetime}_N$(printf "%03d" "$i") --total_num_exps $N &
  COMP_PID="$!"

  cd -

  sleep 2

  # Wait until the evaluation script has finished
  while ps -p $PY_PID > /dev/null
  do
    echo
    echo [LAUNCH_EVALUATION] Sending start navigation command
    echo
    rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" --once
    sleep 2

    # if the current iteration has surpassed the time limit, something went wrong (possibly: [Pipeline]     Bridge failed!). Kill the simulator.
    if ((($(date +%s) - start_time) >= 300))
    then
      echo
      echo
      echo
      echo
      echo "Time limit exceeded. Exiting evaluation script loop."
      echo
      echo
      echo
      echo
      kill -SIGINT $PY_PID
      relaunch_sim=1
      break
    fi

  done

  cat "$SUMMARY_FILE" "./envtest/ros/summary.yaml" > "tmp.yaml"
  mv "tmp.yaml" "$SUMMARY_FILE"

  kill -SIGINT "$COMP_PID"
done


if [ $ROS_PID ]
then
  kill -SIGINT "$ROS_PID"
fi
