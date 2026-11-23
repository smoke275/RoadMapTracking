#include<iostream>

struct ProgressBar
{
    int barWidth = 20;

    void update(double progress)
    {
        if (progress >= 1.0)
        {
            for (int i = 0; i < barWidth; ++i)
                std::cout << "  ";
            done();
            return;
        }

        std::cout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < pos)
                std::cout << "=";
            else if (i == pos)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();
    }

    void done()
    {
        std::cout << "\rDone" << std::endl;
        std::cout.flush();
    }
};