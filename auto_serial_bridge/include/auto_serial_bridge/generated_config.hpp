#pragma once
#include <cstdint>
#include <cstddef>

#include "protocol.h"

namespace auto_serial_bridge {
namespace config {

    constexpr uint32_t DEFAULT_BAUDRATE = 460800;
    constexpr size_t BUFFER_SIZE = 256;
    constexpr uint8_t CFG_FRAME_HEADER1 = 90;
    constexpr uint8_t CFG_FRAME_HEADER2 = 165;

    enum class ChecksumAlgo { NONE, SUM8, XOR8, CRC8 };
    constexpr ChecksumAlgo CHECKSUM_ALGO = ChecksumAlgo::CRC8;

    constexpr bool REQUIRE_HANDSHAKE = false;
    constexpr bool IGNORE_VERSION_MISMATCH = false;
    constexpr bool ENABLE_HEARTBEAT = false;
    constexpr bool STRICT_HEARTBEAT = false;
    constexpr size_t QOS_DEPTH = 10;
    constexpr int HEARTBEAT_TIMEOUT_MS = 0;
    constexpr int RELIABLE_RETRY_INTERVAL_MS = 100;
    constexpr int RELIABLE_MAX_RETRIES = 3;
    constexpr size_t MAX_PACKET_PAYLOAD_SIZE = 12;

    inline constexpr size_t expected_payload_size(PacketID id) {
        switch (id) {
            case PACKET_ID_ACK: return sizeof(Packet_Ack);
            case PACKET_ID_HEARTBEAT: return sizeof(Packet_Heartbeat);
            case PACKET_ID_HANDSHAKE: return sizeof(Packet_Handshake);
            case PACKET_ID_POSEREF: return sizeof(Packet_PoseRef);
            case PACKET_ID_CMDVEL: return sizeof(Packet_CmdVel);
            case PACKET_ID_CAMCMD: return sizeof(Packet_CamCmd);
            case PACKET_ID_STAIRACTIONCMD: return sizeof(Packet_StairActionCmd);
            case PACKET_ID_CAMSIG: return sizeof(Packet_CamSig);
            default: return 0;
        }
    }

    inline constexpr bool is_reliable_packet(PacketID id) {
        switch (id) {
            case PACKET_ID_ACK: return false;
            case PACKET_ID_HEARTBEAT: return false;
            case PACKET_ID_HANDSHAKE: return false;
            case PACKET_ID_POSEREF: return false;
            case PACKET_ID_CMDVEL: return false;
            case PACKET_ID_CAMCMD: return false;
            case PACKET_ID_STAIRACTIONCMD: return false;
            case PACKET_ID_CAMSIG: return false;
            default: return false;
        }
    }

}
}
