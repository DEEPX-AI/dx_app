#include "PacketConvert.h"

using namespace std;

PacketConverter& PacketConverter::operator=(const ds_eth_header_data_t &hdr)
{
  int i;
  unsigned char *pt;

  // dst_addr
  for(i=0; i<6; i++) 
  {
    _m_hdr.dst_addr += to_string(hdr.dst_addr[i]);
    if( i != 5) 
    {
      _m_hdr.dst_addr += ":";
    }
  }

  // src_addr
  for(i=0; i<6; i++) 
  {
    _m_hdr.src_addr += to_string(hdr.src_addr[i]);
    if( i != 5) 
    {
      _m_hdr.src_addr += ":";
    }
  }

  // eth_protocol
  _m_hdr.eth_protocol = to_string(hdr.eth_protocol);

  // ip_ver
  _m_hdr.ip_ver = to_string(hdr.ip_ver);

  // ip_protocol
  _m_hdr.ip_protocol = to_string(hdr.ip_protocol);

  // dst_ip_addr
  pt = (unsigned char*) &hdr.dst_ip_addr;
  for(i=0; i<4; i++)
  {
    _m_hdr.dst_ip_addr += to_string(*pt++);
    _m_hdr.dst_ip_addr += ".";
  }

  // src_ip_addr
  pt = (unsigned char*) &hdr.src_ip_addr;
  for(i=0; i<4; i++)
  {
    _m_hdr.src_ip_addr += to_string(*pt++);
    _m_hdr.src_ip_addr += ".";
  }  

  // sequence
  _m_hdr.sequence = to_string(hdr.sequence);


  // ttl
  _m_hdr.ttl = to_string(hdr.ttl);

  // fag
  _m_hdr.flag = ds_flags[hdr.flag];

  // sevice
  _m_hdr.service = ds_service[hdr.dst_port];

  // payload_length
  _m_hdr.payload_length = to_string(hdr.payload_length);

  // payload
  pt = (unsigned char*) hdr.payload_data;
  for(i=0; i<hdr.payload_length; i++)
  {
    _m_hdr.payload_data += to_string(*pt++);
    _m_hdr.payload_data +=":";
  }
}
