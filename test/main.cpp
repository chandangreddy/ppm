#include <boost/program_options.hpp>
#include <glog/logging.h>

using namespace boost;
#include <iostream>
#include <islutils/common.h>
#include <islutils/cpu.h>
#include <islutils/access_processor.h>

using namespace std;

namespace {
  const size_t ERROR_IN_COMMAND_LINE = 1;
  const size_t SUCCESS = 0;
  const size_t ERROR_UNHANDLED_EXCEPTION = 2;
}

// transform the file called "input" by replacing each scops by 
// the corresponding optimized builder if available. The result is 
// written in a file called "output". "input" and "output" are
// payloads in the "options" struct.

static bool generate_code(struct Options &options) {

  bool res = false;
  switch(options.target) {
    case 1:
      LOG(INFO) << "generating code for CPU" << "\n";
      res = generate_CPU(options);
      break;
    case 2:
      LOG(INFO) << "generating code for GPU" << "\n";
      res = generate_AP(options);
      break;
    //case 3:
    //  res = generate_GPU(options);
    //  break;
    default:
      assert(0 && "options.target not defined");
  }

  return res;
}


int main(int ac, char* av[]) {

  // log on stdout and file.
  google::InitGoogleLogging(av[0]);
  google::SetLogDestination(google::INFO, "/home/parallels/Desktop/INFO.log");
  FLAGS_alsologtostderr = 1; 
  LOG(INFO) << "start logging\n";  

  Options options;

  try {
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "print help message")
      ("input,i", po::value<string>(&options.inputFile),"input file name")
      ("output,o", po::value<string>(&options.outputFile),"output file name")
      ("target,t", po::value<int>(&options.target),"target we generate code for")
      ("function-call,f", po::value<bool>(&options.function_call),"use library calls");

    po::variables_map vm;
    try {
      po::store(po::parse_command_line(ac, av, desc),vm);
      if(vm.count("help")) {
        cout << "command line options" << endl;
        cout << desc << endl;
        return SUCCESS;
      }
      po::notify(vm);
    } catch(po::error& e) {
      std::cerr << "error: " << e.what() << endl;
      std::cerr << desc << endl;
      return ERROR_IN_COMMAND_LINE;
    }

    if(options.target == -1) {
      LOG(INFO) << "target not specified assuming CPU" << "\n";
      options.target = 1;
    }
    if(options.inputFile == "empty") {
      std::cout << "target file not specified.. exit" << std::endl;
      return ERROR_IN_COMMAND_LINE;
    }
    
    bool res = generate_code(options);
    assert(res && "generate_code returned false");

  } catch(std::exception& e) {
    std::cerr << "unhandled excpetion reached main: " << endl;
    std::cerr << e.what() << " exit now " << endl;
    return ERROR_UNHANDLED_EXCEPTION;
  }

  LOG(INFO) << options.inputFile;
  LOG(INFO) << options.outputFile;
  LOG(INFO) << getStringFromTarget(options.target);
  if(!options.function_call) {
    LOG(INFO) << "standard optimizations without function calls";
  } else {
    LOG(INFO) << "optimization based on function calls";
  }

  return SUCCESS;
}
