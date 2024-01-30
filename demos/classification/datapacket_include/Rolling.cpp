#include "Rolling.h"

Rolling::Rolling()
{
	table = new list<map<string, string>>();
	cout << "Rolling window Init" << endl;
}

Rolling::~Rolling()
{
	cout << "Rolling window Bye" << endl;
}

void Rolling::add(string item[7])
/*
* @param item[0] : Source IP
* @param item[1] : Destination IP
* @param item[2] : Source Port
* @param item[3] : Destination Port
* @param item[4] : FLAG
* @param item[5] : PAYLOAD LENGTH
* @param item[6] : LOCAL TIME
*/
{
	/*
	std::tm timestamp = { 58, 20, 14, 24, 3, 123 };

	std::time_t timestamp_seconds = std::mktime(&timestamp);

	auto timestamp_tp = std::chrono::system_clock::from_time_t(timestamp_seconds) +
		std::chrono::milliseconds{ 396000 };
	auto milliseconds_since_epoch = std::chrono::time_point_cast<std::chrono::milliseconds>(timestamp_tp);

	string local_time = to_string(milliseconds_since_epoch.time_since_epoch().count());
	item[6] = local_time;
	*/

	string syn_time = "\0";
	string ack_time = "\0";
	string syn_ack_time = "\0";

	// Data Initialization
	map<string, string> dataToInsert;
	dataToInsert["SRC_IP"] = item[0];
	dataToInsert["DST_IP"] = item[1];
	dataToInsert["SRC_PORT"] = item[2];
	dataToInsert["DST_PORT"] = item[3];
	dataToInsert["FLAG"] = item[4];
	dataToInsert["PAYLOAD_LEN"] = item[5];
	dataToInsert["LOCAL_TIME"] = item[6];
	dataToInsert["SYN_ACK"] = "0";
	dataToInsert["ACK_DAT"] = "0";

	// Data for transaction check 
	pair<string, string> src_ip_pair(item[0], item[1]);
	additional datas;

	if ((item[4].find('S')) != string::npos) {
		syn_time = item[6];
	}

	if ((item[4].find('A')) != string::npos) {
		ack_time = item[6];
	}

	if (((item[4].find('A')) != string::npos) &&
		((item[4].find('S')) != string::npos)) {
		syn_ack_time = item[6];
	}

	auto checkDataExist = transactions.find(src_ip_pair);

	if (checkDataExist == transactions.end()) {
		if (syn_time == "\0") {
			syn_time = "0";
		}
		if (ack_time == "\0") {
			ack_time = "0";
		}
		if (syn_ack_time == "\0") {
			syn_ack_time = "0";
		}

		datas.sbytes = item[5];
		datas.packet_cnt = "1";
	}
	else {
		string sbytes = std::to_string(stoi(checkDataExist->second.sbytes) + stoi(item[5]));
		string packet_cnt = std::to_string(stoi(checkDataExist->second.packet_cnt) + 1);

		if (syn_ack_time != "\0") {
			dataToInsert["SYN_ACK"] = std::to_string(stoll(syn_ack_time) - stoll(transactions.find(src_ip_pair)->second.syn_time));
		}
		if (ack_time != "\0") {
			dataToInsert["ACK_DAT"] = std::to_string(stoll(ack_time) - stoll(transactions.find(src_ip_pair)->second.syn_ack_time));
		}

		if (syn_time == "\0") {
			syn_time = checkDataExist->second.syn_time;
		}
		if (ack_time == "\0") {
			ack_time = checkDataExist->second.ack_time;
		}
		if (syn_ack_time == "\0") {
			syn_ack_time = checkDataExist->second.syn_ack_time;
		}
		datas.sbytes = sbytes;
		datas.packet_cnt = packet_cnt;
	}
	
	datas.syn_time = syn_time;
	datas.ack_time = ack_time;
	datas.syn_ack_time = syn_ack_time;

	transactions[src_ip_pair] = datas;
	table->push_back(dataToInsert);
}

void Rolling::print()
{
	list<map<string, string>>::iterator p;
	map<string, string>::iterator mp;
	map<pair<string, string>, additional>::iterator myIt;

	cout << "==============================================================" << endl;
	cout << "=======================TABLE INFO=============================" << endl;
	for (p = table->begin(); p != table->end(); ++p) {
		cout << "---------------------------------------------------" << endl;
		for (mp = p->begin(); mp != p->end(); ++mp) {
			cout << mp->first << ": " << mp->second << endl;
		}
		cout << "---------------------------------------------------" << endl;
	}
	cout << "==============================================================" << endl;

	cout << "==============================================================" << endl;
	cout << "===================TRANSACTION INFO===========================" << endl;
	for (myIt = transactions.begin(); myIt != transactions.end(); ++myIt) {
		cout << "( " << myIt->first.first << ", " << myIt->first.second << " ) " << endl;
		cout << "sbytes : " << myIt->second.sbytes << endl;
		cout << "packet_cnt : " << myIt->second.packet_cnt << endl;
		cout << "syn_time : " << myIt->second.syn_time << endl;
		cout << "ack_time : " << myIt->second.ack_time << endl;
		cout << "syn_ack_time : " << myIt->second.syn_ack_time << endl;
	}
	cout << "==============================================================" << endl;
}

list<string> Rolling::extractInfo()
{
	list<string> returnList;

	lastComp = table->back();
	currentSRCIP = lastComp.find("SRC_IP")->second;
	currentDSTIP = lastComp.find("DST_IP")->second;

	string syn_ack = lastComp.find("SYN_ACK")->second;
	string ack_dat = lastComp.find("ACK_DAT")->second;
	string spkts = "0";
	string dpkts = "0";
	string sbytes = "0";
	string dbytes = "0";

	// Find data from table and transaction
	pair<string, string> src_ip_pair(currentSRCIP, currentDSTIP);
	pair<string, string> dst_ip_pair(currentDSTIP, currentSRCIP);

	auto fromTransaction = transactions.find(src_ip_pair);

	if (fromTransaction != transactions.end()) {
		spkts = fromTransaction->second.packet_cnt;
		sbytes = fromTransaction->second.sbytes;
	}

	fromTransaction = transactions.find(dst_ip_pair);

	if (fromTransaction != transactions.end()) {
		dpkts = fromTransaction->second.packet_cnt;
		dbytes = fromTransaction->second.sbytes;
	}
	
	returnList.push_back(syn_ack);
	returnList.push_back(ack_dat);
	returnList.push_back(spkts);
	returnList.push_back(dpkts);
	returnList.push_back(sbytes);
	returnList.push_back(dbytes);

	return returnList;
}
