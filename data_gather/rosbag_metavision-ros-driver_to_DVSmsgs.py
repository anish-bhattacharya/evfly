# given a rosbag, read the messages and create a new rosbag with dvs msgs instead

import rosbag, rospy
import sys
sys.path.append('/home/anish/evfly_ws/src/event_array_py')
from event_array_py import Decoder
# manually include path
# sys.path.append('/home/anish/Downloads/event_array_py-master/build')
from dvs_msgs.msg import EventArray, Event
import numpy.lib.recfunctions as rf

# rosmsg show dvs_msgs/EventArray
# std_msgs/Header header
#   uint32 seq
#   time stamp
#   string frame_id
# uint32 height
# uint32 width
# dvs_msgs/Event[] events
#   uint16 x
#   uint16 y
#   time ts
#   bool polarity


topic = '/event_camera/events'
input_bag_path = sys.argv[1]

decoder = Decoder()

n = 0

# note the 't' last element of each event is actually the absolute sensor timestamp
# if I want to use the ros timestamp of the events (for syncing with the depth images), can I take the EventArray ros time and add the differential event time from the first event onwards? I.e. I'll subtract each event's t by the first event's t.

# rosmsg times are really not accurate.
# use the first ros msg time as the base time, and then on add the differential sensor timestamp

# event structure: [x, y, p, t]

first_rosmsg_timestamp = None
first_rosbag_timestamp = None
first_sensor_timestamp = None

output_bag_path = sys.argv[2]
# Open the output bag file for writing
with rosbag.Bag(output_bag_path, 'w') as outbag:
    # Open the input bag file for reading
    with rosbag.Bag(input_bag_path, 'r') as inbag:
        # Iterate through each message in the input bag
        for topic, msg, t in inbag.read_messages(topics=topic):

            if first_rosbag_timestamp is None:
                first_rosbag_timestamp = t.to_nsec()
                print(f'First event array msg rosbag timestamp (ns): {first_rosbag_timestamp}')

            if first_rosmsg_timestamp is None:
                first_rosmsg_timestamp = msg.header.stamp.to_nsec()
                print(f'First event array msg ros timestamp (ns): {first_rosmsg_timestamp}')

            # # DEBUG: early stopping to test
            # if (t.to_nsec()-first_rosbag_timestamp)/1e9 > 5:
            #     break

            # Decode the message
            decoder.decode_bytes(msg.encoding, msg.width, msg.height,
                                 msg.time_base, msg.events)
            cd_events = decoder.get_cd_events()
            cd_events = rf.structured_to_unstructured(cd_events)

            if first_sensor_timestamp is None:
                first_sensor_timestamp = cd_events[0][-1] * 1e3
                print(f'First event array batch sensor timestamp (ns): {first_sensor_timestamp}')

            # every 100 messages, print rosbag timestamp and the bag's total time
            if n % 100 == 0:
                print(f'At rosbag time {(t.to_nsec()-first_rosbag_timestamp)/1e9:.6f} / {inbag.get_end_time() - inbag.get_start_time():.6f}, processing {cd_events.shape[0]} events at this timestamp.')

            # print(cd_events.shape)

            # break

            # # compute time from ros msg header and convert to nsec
            # base_time = msg.header.stamp.to_nsec()
            # event_time = base_time + []

            # fill a dvs_msgs EventArray with this batch of events
            event_array = EventArray()
            event_array.header = msg.header
            event_array.height = msg.height
            event_array.width = msg.width
            event_array.events = []
            for ev_i in range(cd_events.shape[0]):
                ev = Event()
                ev.x = cd_events[ev_i][0]
                ev.y = cd_events[ev_i][1]
                ev.polarity = cd_events[ev_i][2]

                # setting timestamp of each event
                
                # first_rosmsg_timestamp is the timestamp roughly at the end of the event array batch since it needs to publish out the message
                ns_t = (cd_events[ev_i][3] * 1e3 - first_sensor_timestamp) + first_rosmsg_timestamp
                ev.ts = rospy.Time.from_sec(ns_t/1e9)

                # # make up timestamps but put each message in order, roughly equal spacing, in the first 1/500 of a second since the message rate is 250Hz
                # dt = 1./500. / cd_events.shape[0]
                # ev.ts = rospy.Time.from_sec(event_array.header.stamp.to_sec() + ev_i * dt)

                event_array.events.append(ev)

            # Write the message to the output bag to topic '/capture_node/events'
            outbag.write('/capture_node/events', event_array, t)

            n += 1
