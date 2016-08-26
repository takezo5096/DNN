/*
 * word_embed.h
 *
 *  Created on: 2016/02/05
 *      Author: takeshi.fujita
 */

#ifndef WORD_EMBED_H_
#define WORD_EMBED_H_

#include <string>
#include <sstream>
#include "tokenizer.h"

#define IDX2F(i,j,ld) ((((j))*(ld))+((i)))


using namespace std;

class WordEmbed {

    Tokenizer token;

    vector<int> dataset;
    map<string, int> idmap;

public:
    WordEmbed(){

    }

    std::string replace( std::string String1, std::string String2, std::string String3 )
    {
        std::string::size_type  Pos( String1.find( String2 ) );

        while( Pos != std::string::npos )
        {
            String1.replace( Pos, String2.length(), String3 );
            Pos = String1.find( String2, Pos + String3.length() );
        }

        return String1;
    }

    std::vector<std::string> split(const std::string &str, char sep)
    {
        std::vector<std::string> v;
        std::stringstream ss(str);
        std::string buffer;
        while( std::getline(ss, buffer, sep) ) {
            v.push_back(buffer);
        }
        return v;
    }

    void add(string sentence, bool tokenize){

        //cout << sentence << endl;
        vector<string> words;
        if (tokenize) words = token.parse(sentence);
        else words = split(sentence, ' ');

        for(string w : words){
            if (idmap.count(w) == 0){
                idmap[w] = idmap.size();
            }
            dataset.push_back(idmap[w]);
        }
    }

    int getVocabSize(){
        return idmap.size();
    }

    vector<int> getDataset(){
        return dataset;
    }

    void toOneHot(float *data, int id, int col){

        int vocab_size=getVocabSize();
        for(int i=0; i<vocab_size; i++){
            if (i==id) data[IDX2F(i, col, vocab_size)] = 1.;
            else data[IDX2F(i, col, vocab_size)] = 0.;
        }
    }



    void toOneHots(float *data, int start, int end){

        int j=0;
        for(int i=start; i<end; i++){
            toOneHot(data, dataset[i], j);
            j++;
        }

    }
};



#endif /* WORD_EMBED_H_ */
