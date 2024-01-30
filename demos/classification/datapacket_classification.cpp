#include "datapacket_classification.h"
/* DEEPX */
#include "dxrt/dxrt_api.h"

#include <string.h>
#include <getopt.h>
#include <mutex>

static struct option const opts[] = {
	{ "model", required_argument, 0, 'm' },
	{ "bin",   required_argument, 0, 'b' },
	{ "async", no_argument,       0, 'a' },
	{ "help",  no_argument,       0, 'h' },
	{ 0, 0, 0, 0 }
};
const char* usage =
"network packet classification demo\n"
"  -m, --model   define model path\n"
"  -a, --async   asynchronous inference\n"
"  -b, --bin     use binary file input with preprocessed data\n"
"  -h, --hazard	 enable hazard mode\n"
;
void help()
{
	cout << usage << endl;
}

/* Data of the user interface - only for async mode */
typedef struct packet_interface_user_t {
	string 	port_id;
	int		number;
} packet_interface_user;
static vector <packet_interface_user> g_user_args;

////////////////// Test Packet ///////////////////////
/* Pre-processing*/
eth_header_data_t pNetHeader = {
	"11111", // dst_addr
	"11111", // src_addr
	"3", // eth_protocol 
	"4", // ip_ver
	"6",  // ip_protocol
	"11111647", // dst-ip_addr
	"11111674", // src_ip_addr 
	"2987", // dst_port
	"1234", // src_port
	"2", // sequence
	"", // local_time
	"32", // payload_length
	"SA", // flag
	"23", // ttl
	"HTTP", // service
	"17:03:03:00:4a:37:d4:a9:eb:8f:da:62:1b:57:7c:55:c2:9f:7d:c4:3f:38:70:3e:13:5f:ec:12:cb:66:d0:5b:93:43:49:06:ab:30:9f:c7:83:c6:d3:01:76:c1:d2:c7:9a:22:33:c2:49:b8:7d:10:9e:d2:a2:d7:ff:ea:08:79:3e:d9:a8:f8:79:5e:84:1d:e1:cc:19:ae:ae:96:b2"
};

int main(int argc, char *argv[])
{
	int optCmd, loops=1;
	string modelPath="", binFile="";
	bool asyncInference = false;
	bool hazardMode = false;

	if(argc==1)
	{
		cout << "Error: no arguments." << endl;
		help();
		return -1;
	}
	while ((optCmd = getopt_long(argc, argv, "m:ab:h", opts,
		NULL)) != -1) {
		switch (optCmd) {
			case '0':
				break;
			case 'm':
				modelPath = strdup(optarg);
				break;
			case 'a':
				asyncInference = true;
				break;
			case 'b':
				binFile = strdup(optarg);
				break;
			case 'h':
				hazardMode = true;
				break;
			default:
				help();
				exit(0);
				break;
		}
	}
	LOG_VALUE(modelPath);
	LOG_VALUE(asyncInference);
    LOG_VALUE(binFile);
	LOG_VALUE(hazardMode);
	if(modelPath.empty())
	{
		cout << "Error: no model argument." << endl;
		help();
		return -1;
	}

	/* Binary input data with preprocessed data */
	if(!binFile.empty())
	{
		auto ie = dxrt::InferenceEngine(modelPath);
		FILE *fp = fopen(binFile.c_str(), "rb");
		if (fp == NULL)
		{
			cout << "File open error! (" << binFile << ")" << endl;
			return -1;
		}
		auto inputSize = ie.input_size();
		int8_t* data = new int8_t[inputSize];
		

		fread(data, inputSize, 1, fp);
		fclose(fp);

		/* inference */
		auto outputs = ie.Run(data);
		int npuId = 0;
		auto result = *(uint16_t*)outputs.front()->data();
		cout << "[Packet Classification] result :: " << result <<" [0:normal packet, 1:attack packet]" << endl;
		return 0;
	}

	/* Load Model */
    auto& profiler = dxrt::Profiler::GetInstance();// for time measurement
	auto ie = dxrt::InferenceEngine(modelPath);

	/* Register the post-processing callback function */
		int callBackCnt = -1;
		static mutex lk;
		std::function<int(vector<shared_ptr<dxrt::Tensor>>, void*)> postProcCallBack = \
			[&](vector<shared_ptr<dxrt::Tensor>> outputs, void* args)
			{
				{
					/* Restore raw frame index from tensor */
					lk.lock();
					/* Get Classification Result, and Post-processing */
					int npuId = 0;
					auto result = outputs.front()->data();
					/* Parsing user-interface */
					if (g_user_args.size() > 0)
					{
						packet_interface_user *user_args = (packet_interface_user *) args;
						cout << "User args string : " <<  user_args->port_id <<
							", index:" << user_args->number << endl;
					}
					cout << "[Packet Classification] result :: " << result <<" [0:normal packet, 1:attack packet]" << endl;
					lk.unlock();
				}
				return 0;
			};
		ie.RegisterCallBack(postProcCallBack);

	/* Pre-processing */
	void * input = nullptr;
	if (hazardMode == false)
	{
		for (int i=0; i<49; i++)
		{
			input = PreProcessor(&pNetHeader, hazardMode);
		}
	}
	else
	{
		/* Set user-interface */
		static int index = 0;
		packet_interface_user user_args = {"test", index++};
		g_user_args.emplace_back(user_args);

		input = PreProcessor(&pNetHeader, hazardMode);
	}

	/* Inference */
	if (asyncInference)
	{
		int reqId = ie.RunAsync(input, (void*)&g_user_args.back());
		ie.Wait(reqId);
	}
	else
	{
		auto outputs = ie.Run(input);
		int npuId = 0;
		auto result = outputs.front()->data();
		cout << "[Packet Classification] result :: " << result <<" [0:normal packet, 1:attack packet]" << endl;
	}

	return 0;
}
