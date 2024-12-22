#include "ros/ros.h"
// #include "dvs_msgs/Event.h"
// #include "dvs_msgs/EventArray.h"
#include "prophesee_event_msgs/Event.h"
#include "prophesee_event_msgs/EventArray.h"
#include "std_msgs/UInt8MultiArray.h"

class ImagePublisher {
public:
    ImagePublisher() : nh_("~"), image_data_(IMAGE_WIDTH * IMAGE_HEIGHT, 128) {
        // Initialize the ROS node
        ros::NodeHandle nh;

        // Subscribe to the 'EventArray' topic with the master
        sub_ = nh.subscribe("/prophesee/camera/cd_events_buffer", 1, &ImagePublisher::eventArrayCallback, this);

        // Create a publisher for the image
        image_pub_ = nh_.advertise<std_msgs::UInt8MultiArray>("/output/proc_evs", 1);

        // Set up a timer to reset and publish the image at PUBLISH_RATE
        timer_ = nh.createTimer(ros::Duration(1.0 / PUBLISH_RATE), &ImagePublisher::timerCallback, this);
    }

    void eventArrayCallback(const prophesee_event_msgs::EventArray::ConstPtr& msg) {
        ROS_INFO("Current ROS Time: %f", ros::Time::now().toSec());
        ROS_INFO("Header Time of Received Message: %f", msg->header.stamp.toSec());

        // Iterate through each event and update the image
        for (const auto& event : msg->events) {
            // Check if the event coordinates are within the image bounds
            if (event.x < IMAGE_WIDTH && event.y < IMAGE_HEIGHT) {
                // Increment the event count at the corresponding pixel
                if (event.polarity == true) {
                    image_data_[event.y * IMAGE_WIDTH + event.x]++;
                } else {
                    image_data_[event.y * IMAGE_WIDTH + event.x]--;
                }
            }
        }
    }

    void timerCallback(const ros::TimerEvent&) {
        // Publish the current image data
        std_msgs::UInt8MultiArray array_msg;
        array_msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
        array_msg.layout.dim[0].label = "data";
        array_msg.layout.dim[0].size = image_data_.size();
        array_msg.layout.dim[0].stride = image_data_.size();
        array_msg.layout.data_offset = 0;

        // Copy data from vector to the message
        array_msg.data = image_data_;

        // Publish the flattened data
        image_pub_.publish(array_msg);

        // Reset the image data for the next frame
        std::fill(image_data_.begin(), image_data_.end(), 128);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher image_pub_;
    ros::Timer timer_;
    std::vector<uint8_t> image_data_;

    static const int IMAGE_WIDTH = 640;
    static const int IMAGE_HEIGHT = 480;
    static const int PUBLISH_RATE = 30;
};

int main(int argc, char **argv) {
    // Initialize the ROS node
    ros::init(argc, argv, "image_creator_node");

    ImagePublisher image_publisher;

    // Spin, waiting for messages
    ros::spin();

    return 0;
}