#pragma once

#include <string>
using namespace std;


typedef struct tagEthHeaderDataType {

	string   dst_addr;
	string   src_addr; 
	string   eth_protocol;
	string   ip_ver; 
	
	// Require
	string    ip_protocol; // need (UDP, TCP, ...
	string    dst_ip_addr; // need (172.163.222.34)
	string    src_ip_addr; // need (172.163.222.34)
	string    dst_port; // need (9999)
	string    src_port; // need (1232)

	// Require end
	string    sequence;
	string    local_time;
	string    payload_length; // need (99)

	// Require
	string    flag; // need (SYN -> S, ACK -> A, SYNACK -> SA ... etc)
	string    ttl; // need source to destination time to live
	string    service; // need (HTTP, stp, smtp.. etc) when service doesn't exist, please send'-'
	
	// Require end
	string    payload_data; // need

} eth_header_data_t;
