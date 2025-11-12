#include <iostream>
#include "trtyolo.hpp"
#include <optional>
int main()
{
    try
    {
        std::string engine_path = "/home/lyk/TensorRT-YOLO/examples/detect/models/ball.engine";

        trtyolo::InferOption option;
        option.enableSwapRB();
        option.setNormalizeParams({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});

        std::cout << "[TEST] Create YOLO model..." << std::endl;
        trtyolo::DetectModel model(engine_path, option);
        std::cout << "[TEST] YOLO model created." << std::endl;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Error] " << e.what() << std::endl;
        return -1;
    }
}
