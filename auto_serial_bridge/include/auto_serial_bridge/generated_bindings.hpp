#pragma once
#include <cstdint>
#include <cstring>
#include <functional>
#include "auto_serial_bridge/serial_controller.hpp"
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/u_int32.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include "protocol.h"

namespace auto_serial_bridge {
namespace generated {

template <typename T> void register_subscriber(SerialController* node, const std::string& topic, PacketID id);

inline void register_all(SerialController* node) {
    // Ack (ROS -> MCU)
    node->add_subscription(node->create_subscription<std_msgs::msg::Int32MultiArray>(
        "/task/ack", 10,
        [node](const std_msgs::msg::Int32MultiArray::SharedPtr msg, const rclcpp::MessageInfo& msg_info) {
            if (node->should_skip_loopback(PACKET_ID_ACK, msg_info)) {
                return;
            }
            if (msg->data.size() < 2) {
                RCLCPP_ERROR_THROTTLE(
                    node->get_logger(), *node->get_clock(), 2000,
                    "Message for Ack on /task/ack requires at least 2 entries in data, got %zu",
                    msg->data.size());
                return;
            }
            Packet_Ack pkt;
            if (msg->data[0] < 0 || msg->data[0] > 255) {
                RCLCPP_ERROR_THROTTLE(
                    node->get_logger(), *node->get_clock(), 2000,
                    "Message for Ack on /task/ack field acked_id is out of uint8 range [0, 255]: %d",
                    static_cast<int>(msg->data[0]));
                return;
            }
            pkt.acked_id = static_cast<uint8_t>(msg->data[0]);
            if (msg->data[1] < 0 || msg->data[1] > 255) {
                RCLCPP_ERROR_THROTTLE(
                    node->get_logger(), *node->get_clock(), 2000,
                    "Message for Ack on /task/ack field ack_seq is out of uint8 range [0, 255]: %d",
                    static_cast<int>(msg->data[1]));
                return;
            }
            pkt.ack_seq = static_cast<uint8_t>(msg->data[1]);
            node->send_packet(PACKET_ID_ACK, pkt);
            static bool has_previous_pkt_Ack = false;
            static Packet_Ack previous_pkt_Ack{};
            const bool should_log_Ack = !has_previous_pkt_Ack || std::memcmp(&previous_pkt_Ack, &pkt, sizeof(Packet_Ack)) != 0;
            if (should_log_Ack) {
                previous_pkt_Ack = pkt;
                has_previous_pkt_Ack = true;
                RCLCPP_DEBUG(node->get_logger(), "TX Ack: acked_id=%u, ack_seq=%u", static_cast<unsigned int>(pkt.acked_id), static_cast<unsigned int>(pkt.ack_seq));
            }
        }));

    // Heartbeat (ROS -> MCU)
    node->add_subscription(node->create_subscription<std_msgs::msg::UInt32>(
        "/task/heartbeat", 10,
        [node](const std_msgs::msg::UInt32::SharedPtr msg, const rclcpp::MessageInfo& msg_info) {
            if (node->should_skip_loopback(PACKET_ID_HEARTBEAT, msg_info)) {
                return;
            }
            Packet_Heartbeat pkt;
            pkt.count = msg->data;
            node->send_packet(PACKET_ID_HEARTBEAT, pkt);
            static bool has_previous_pkt_Heartbeat = false;
            static Packet_Heartbeat previous_pkt_Heartbeat{};
            const bool should_log_Heartbeat = !has_previous_pkt_Heartbeat || std::memcmp(&previous_pkt_Heartbeat, &pkt, sizeof(Packet_Heartbeat)) != 0;
            if (should_log_Heartbeat) {
                previous_pkt_Heartbeat = pkt;
                has_previous_pkt_Heartbeat = true;
                RCLCPP_DEBUG(node->get_logger(), "TX Heartbeat: count=%u", static_cast<unsigned int>(pkt.count));
            }
        }));

    // Handshake (ROS -> MCU)
    node->add_subscription(node->create_subscription<std_msgs::msg::UInt32>(
        "/task/handshake", 10,
        [node](const std_msgs::msg::UInt32::SharedPtr msg, const rclcpp::MessageInfo& msg_info) {
            if (node->should_skip_loopback(PACKET_ID_HANDSHAKE, msg_info)) {
                return;
            }
            Packet_Handshake pkt;
            pkt.protocol_hash = msg->data;
            node->send_packet(PACKET_ID_HANDSHAKE, pkt);
            static bool has_previous_pkt_Handshake = false;
            static Packet_Handshake previous_pkt_Handshake{};
            const bool should_log_Handshake = !has_previous_pkt_Handshake || std::memcmp(&previous_pkt_Handshake, &pkt, sizeof(Packet_Handshake)) != 0;
            if (should_log_Handshake) {
                previous_pkt_Handshake = pkt;
                has_previous_pkt_Handshake = true;
                RCLCPP_DEBUG(node->get_logger(), "TX Handshake: protocol_hash=%u", static_cast<unsigned int>(pkt.protocol_hash));
            }
        }));

    // PoseRef (ROS -> MCU)
    node->add_subscription(node->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/serial/pose_ref", 10,
        [node](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
            if (msg->data.size() < 3) {
                RCLCPP_ERROR_THROTTLE(
                    node->get_logger(), *node->get_clock(), 2000,
                    "Message for PoseRef on /serial/pose_ref requires at least 3 entries in data, got %zu",
                    msg->data.size());
                return;
            }
            Packet_PoseRef pkt;
            pkt.x_map = msg->data[0];
            pkt.y_map = msg->data[1];
            pkt.yaw_map = msg->data[2];
            node->send_packet(PACKET_ID_POSEREF, pkt);
        }));

    // CmdVel (ROS -> MCU)
    node->add_subscription(node->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", 10,
        [node](const geometry_msgs::msg::Twist::SharedPtr msg) {
            Packet_CmdVel pkt;
            pkt.linear_x = msg->linear.x;
            pkt.linear_y = msg->linear.y;
            pkt.angular_z = msg->angular.z;
            node->send_packet(PACKET_ID_CMDVEL, pkt);
            static bool has_previous_pkt_CmdVel = false;
            static Packet_CmdVel previous_pkt_CmdVel{};
            const bool should_log_CmdVel = !has_previous_pkt_CmdVel || std::memcmp(&previous_pkt_CmdVel, &pkt, sizeof(Packet_CmdVel)) != 0;
            if (should_log_CmdVel) {
                previous_pkt_CmdVel = pkt;
                has_previous_pkt_CmdVel = true;
                RCLCPP_DEBUG(node->get_logger(), "TX CmdVel: linear_x=%.3f, linear_y=%.3f, angular_z=%.3f", static_cast<double>(pkt.linear_x), static_cast<double>(pkt.linear_y), static_cast<double>(pkt.angular_z));
            }
        }));

    // CamCmd (ROS -> MCU)
    node->add_subscription(node->create_subscription<std_msgs::msg::UInt8>(
        "/camera/view_cmd", 10,
        [node](const std_msgs::msg::UInt8::SharedPtr msg) {
            Packet_CamCmd pkt;
            pkt.camaction = msg->data;
            node->send_packet(PACKET_ID_CAMCMD, pkt);
            static bool has_previous_pkt_CamCmd = false;
            static Packet_CamCmd previous_pkt_CamCmd{};
            const bool should_log_CamCmd = !has_previous_pkt_CamCmd || std::memcmp(&previous_pkt_CamCmd, &pkt, sizeof(Packet_CamCmd)) != 0;
            if (should_log_CamCmd) {
                previous_pkt_CamCmd = pkt;
                has_previous_pkt_CamCmd = true;
                RCLCPP_DEBUG(node->get_logger(), "TX CamCmd: camaction=%u", static_cast<unsigned int>(pkt.camaction));
            }
        }));

    // StairActionCmd (ROS -> MCU)
    node->add_subscription(node->create_subscription<std_msgs::msg::UInt8>(
        "/stair_action_cmd", 10,
        [node](const std_msgs::msg::UInt8::SharedPtr msg) {
            Packet_StairActionCmd pkt;
            pkt.action = msg->data;
            node->send_packet(PACKET_ID_STAIRACTIONCMD, pkt);
            static bool has_previous_pkt_StairActionCmd = false;
            static Packet_StairActionCmd previous_pkt_StairActionCmd{};
            const bool should_log_StairActionCmd = !has_previous_pkt_StairActionCmd || std::memcmp(&previous_pkt_StairActionCmd, &pkt, sizeof(Packet_StairActionCmd)) != 0;
            if (should_log_StairActionCmd) {
                previous_pkt_StairActionCmd = pkt;
                has_previous_pkt_StairActionCmd = true;
                RCLCPP_DEBUG(node->get_logger(), "TX StairActionCmd: action=%u", static_cast<unsigned int>(pkt.action));
            }
        }));

    // CamSig (ROS -> MCU)
    node->add_subscription(node->create_subscription<std_msgs::msg::UInt8>(
        "/camera/view_sog", 10,
        [node](const std_msgs::msg::UInt8::SharedPtr msg) {
            Packet_CamSig pkt;
            pkt.camsignal = msg->data;
            node->send_packet(PACKET_ID_CAMSIG, pkt);
            static bool has_previous_pkt_CamSig = false;
            static Packet_CamSig previous_pkt_CamSig{};
            const bool should_log_CamSig = !has_previous_pkt_CamSig || std::memcmp(&previous_pkt_CamSig, &pkt, sizeof(Packet_CamSig)) != 0;
            if (should_log_CamSig) {
                previous_pkt_CamSig = pkt;
                has_previous_pkt_CamSig = true;
                RCLCPP_DEBUG(node->get_logger(), "TX CamSig: camsignal=%u", static_cast<unsigned int>(pkt.camsignal));
            }
        }));

}

struct ProtocolPublishers {
    rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr pub_Ack;
    rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr pub_Heartbeat;
    rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr pub_Handshake;

    void init(SerialController* node) {
        pub_Ack = node->create_publisher<std_msgs::msg::Int32MultiArray>("/task/ack", 10);
        node->register_loopback_publisher(PACKET_ID_ACK, pub_Ack);
        pub_Heartbeat = node->create_publisher<std_msgs::msg::UInt32>("/task/heartbeat", 10);
        node->register_loopback_publisher(PACKET_ID_HEARTBEAT, pub_Heartbeat);
        pub_Handshake = node->create_publisher<std_msgs::msg::UInt32>("/task/handshake", 10);
        node->register_loopback_publisher(PACKET_ID_HANDSHAKE, pub_Handshake);
    }
};

inline void dispatch_packet(ProtocolPublishers& pubs, uint8_t id, const std::vector<uint8_t>& data, const rclcpp::Logger& logger) {
    switch(id) {
        case PACKET_ID_ACK: {
            if (data.size() != sizeof(Packet_Ack)) break;
            const Packet_Ack* pkt = reinterpret_cast<const Packet_Ack*>(data.data());
            auto msg = std_msgs::msg::Int32MultiArray();
            msg.data.resize(2);
            msg.data[0] = static_cast<int32_t>(pkt->acked_id);
            msg.data[1] = static_cast<int32_t>(pkt->ack_seq);
            static bool has_previous_pkt_Ack = false;
            static Packet_Ack previous_pkt_Ack{};
            const bool should_log_Ack = !has_previous_pkt_Ack || std::memcmp(&previous_pkt_Ack, pkt, sizeof(Packet_Ack)) != 0;
            if (should_log_Ack) {
                previous_pkt_Ack = *pkt;
                has_previous_pkt_Ack = true;
                RCLCPP_DEBUG(logger, "RX Ack: acked_id=%u, ack_seq=%u", static_cast<unsigned int>(pkt->acked_id), static_cast<unsigned int>(pkt->ack_seq));
            }
            if (pubs.pub_Ack) {
                pubs.pub_Ack->publish(msg);
            }
            break;
        }
        case PACKET_ID_HEARTBEAT: {
            if (data.size() != sizeof(Packet_Heartbeat)) break;
            const Packet_Heartbeat* pkt = reinterpret_cast<const Packet_Heartbeat*>(data.data());
            auto msg = std_msgs::msg::UInt32();
            msg.data = pkt->count;
            static bool has_previous_pkt_Heartbeat = false;
            static Packet_Heartbeat previous_pkt_Heartbeat{};
            const bool should_log_Heartbeat = !has_previous_pkt_Heartbeat || std::memcmp(&previous_pkt_Heartbeat, pkt, sizeof(Packet_Heartbeat)) != 0;
            if (should_log_Heartbeat) {
                previous_pkt_Heartbeat = *pkt;
                has_previous_pkt_Heartbeat = true;
                RCLCPP_DEBUG(logger, "RX Heartbeat: count=%u", static_cast<unsigned int>(pkt->count));
            }
            if (pubs.pub_Heartbeat) {
                pubs.pub_Heartbeat->publish(msg);
            }
            break;
        }
        case PACKET_ID_HANDSHAKE: {
            if (data.size() != sizeof(Packet_Handshake)) break;
            const Packet_Handshake* pkt = reinterpret_cast<const Packet_Handshake*>(data.data());
            auto msg = std_msgs::msg::UInt32();
            msg.data = pkt->protocol_hash;
            static bool has_previous_pkt_Handshake = false;
            static Packet_Handshake previous_pkt_Handshake{};
            const bool should_log_Handshake = !has_previous_pkt_Handshake || std::memcmp(&previous_pkt_Handshake, pkt, sizeof(Packet_Handshake)) != 0;
            if (should_log_Handshake) {
                previous_pkt_Handshake = *pkt;
                has_previous_pkt_Handshake = true;
                RCLCPP_DEBUG(logger, "RX Handshake: protocol_hash=%u", static_cast<unsigned int>(pkt->protocol_hash));
            }
            if (pubs.pub_Handshake) {
                pubs.pub_Handshake->publish(msg);
            }
            break;
        }
    }
}
}
}
