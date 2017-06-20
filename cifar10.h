//
// Created by 藤田 毅 on 2017/03/02.
//

#ifndef DNN_CIFAR10_H
#define DNN_CIFAR10_H

class CIFAR10 {

public:

    const int number_of_images = 10000;
    int channel = 3;
    int rows = 32*32;
    int rgb_data_size = rows * channel;

    vector<vector<float> > images;
    vector<float> labels;

    void readFile(string filename){
        ifstream ifs(filename, std::ios::in | std::ios::binary);

        for (int i = 0; i < number_of_images; i++) {

            unsigned char label_num;
            unsigned char imgc;
            vector<float> img_f;

            //read label
            ifs.read((char *) &label_num, sizeof(unsigned char));
            labels.push_back((float) label_num);

            //read data
            for (int k = 0; k < rgb_data_size; k++) {
                ifs.read((char *)&imgc, sizeof(unsigned char));
                img_f.push_back((float) imgc);
            }
            images.push_back(img_f);
        }

        ifs.close();
    }


    vector<vector<float> > getDatas(){
        return images;
    }

    vector<float> getLabels(){
        return labels;
    }

};


#endif //DNN_CIFAR10_H
