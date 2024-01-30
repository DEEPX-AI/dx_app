#ifndef PACKET_CONVERT_H
#define PACKET_CONVERT_H
#include <map>

using namespace std;

#include "TagEthHeaderDataType.h"

typedef struct ds_tagEthHeaderDataType {

  char    dst_addr[6];
  char    src_addr[6];
  short   eth_protocol;
  char    ip_ver;
  char    ip_protocol;
  unsigned int dst_ip_addr;
  unsigned int src_ip_addr;
  short dst_port;
  short src_port;
  int     sequence;
  short   ttl;
  short   flag;
  long long  local_time;
  unsigned int payload_length;
  void    *payload_data;

} ds_eth_header_data_t;


class PacketConverter
{

  public:
    PacketConverter() {};
    ~PacketConverter() {};

  public:
    virtual PacketConverter& operator=(const ds_eth_header_data_t &hdr);
    tagEthHeaderDataType& getHdr() { return _m_hdr; }

  private:
    tagEthHeaderDataType _m_hdr;    

  private:
    std::map<int, string> ds_service =
    {
      {80, "HTTP"},
      {25, "SMTP"}, {465, "SMTP"}, {567, "SMTP"}
    };

    std::map<int, string> ds_flags =
    {
      
      {0x00, ""}, {0x01, "F"}, {0x02, "S"}, {0x03, "FS"},
      {0x08, "P"}, {0x09, "FP"}, {0x0A, "SP"}, {0x0B, "FSP"}, 
      {0x10, "A"}, {0x11, "FA"}, {0x12, "SA"}, {0x13, "FSA"},

      {0x14, "RA"}, {0x18, "PA"}, {0x19, "FPA"}, {0x1A, "SPA"},
      {0x1B, "FSPA"}, {0x40, "E"}, {0x41, "FE"}, {0x42, "SE"},
      {0x43, "FSE"}, {0x48, "PE"}, {0x49, "FPE"}, {0x4A, "SPE"},

      {0x4B, "FSPE"}, {0x50, "AE"}, {0x51, "FAE"}, {0x52, "SAE"},
      {0x53, "FSAE"}, {0x58, "PAE"}, {0x59, "FPAE"}, {0x5A, "SPAE"},

      {0x5B, "FSPAE"}, {0x80, "C"}, {0x81, "FC"}, {0x82, "SC"},
      {0x83, "FPC"}, {0x88, "PC"}, {0x89, "FPC"}, {0x8A, "SPC"},

      {0x8B, "FSPC"}, {0x90, "AC"}, {0x91, "FAC"}, {0x92, "SAC"},
      {0x93, "FSAC"}, {0x98, "PAC"}, {0x99, "FPAC"}, {0x9A, "SPAC"},
      {0x9B, "FSPAC"}, {0xC0, "EC"}, {0xC1, "FEC"}, {0xC2, "SEC"},
 
      {0xC3, "FSEC"}, {0xC8, "PEC"}, {0xC9, "FPEC"}, {0xCA, "SPEC"},
      {0xCB, "FSPEC"}, {0xD0, "AEC"}, {0xD1, "FAEC"}, {0xD2, "SAEC"},
      {0xD3, "FSAEC"}, {0xD8, "PAEC"}, {0xD9, "FPAEC"}, {0xDA, "SPAEC"},
      {0xDB, "FSPAEC"}

    };
};

#endif  // PACKET_CONVERT_H
