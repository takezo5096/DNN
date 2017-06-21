/*
 * word_embed.h
 *
 */

#ifndef WORD_EMBED_H_
#define WORD_EMBED_H_

#include <string>
#include <sstream>
#include "tokenizer.h"

#define IDX2F(i,j,ld) ((((j))*(ld))+((i)))


using namespace std;

typedef pair<string, int> ass_arr;
bool sort_less(const ass_arr& left,const ass_arr& right){
    return left.second < right.second;
}
bool sort_greater(const ass_arr& left,const ass_arr& right){
    return left.second > right.second;
}

class WordEmbed {

    Tokenizer token;

    vector<vector<string>> sequences;
    vector<vector<int>> sequences_ids;
    map<string, int> idmap;
    map<int, string> idmap_reverse;

    map<string, int> words_count;



    int vocab_size = 0;

public:

    const int UNK_ID = 0;
    const int SOS_ID = 1;
    const int EOS_ID = 2;
    const int PAD_ID = 3;

    WordEmbed(int vocab_size){

        this->vocab_size = vocab_size;

        idmap["<unk>"] = UNK_ID;
        idmap_reverse[UNK_ID] = "<unk>";
        idmap["<sos>"] = SOS_ID;
        idmap_reverse[SOS_ID] = "<sos>";
        idmap["<eos>"] = EOS_ID;
        idmap_reverse[EOS_ID] = "<eos>";
        idmap["<pad>"] = PAD_ID;
        idmap_reverse[PAD_ID] = "<pad>";
    }


    int getWordCount(){
        return words_count.size();
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


    void addSentences(vector<string> seqs, bool tokenize, bool addEOS){
        for (auto s : seqs){
            add(s, tokenize, addEOS);
        }

        vector<pair<string, int> > pairs(words_count.size());

        int i=0;
        for (auto v : words_count){
            pairs[i] = make_pair(v.first, v.second);
            i++;
        }

        sort(pairs.begin(),pairs.end(),sort_greater);

        int cnt = 0;
        for(auto v : pairs){
            string w = v.first;
            if (idmap.count(w) == 0){
                int id = idmap.size();
                idmap[w] = id;
                idmap_reverse[id] = w;
            }
            if (cnt == vocab_size) break;
            cnt++;
        }

        for (int i=0; i<sequences.size(); i++){
            vector<string> words = sequences[i];
            vector<int> word_ids;
            for (int j=0; j<words.size(); j++){
                if (idmap.count(words[j]) == 0)word_ids.push_back(UNK_ID);
                else word_ids.push_back(idmap[words[j]]);
            }
            sequences_ids.push_back(word_ids);

        }

    }

    void add(string sentence, bool tokenize, bool addEOS){

        if (sentence == "") return;


        vector<string> words, words_final;
        if (tokenize) words = token.parse(sentence);
        else words = split(sentence, ' ');

        if (addEOS) words.push_back("<eos>");

        for (auto w : words){
            if (w != "") words_final.push_back(w);
        }

        for (auto w : words_final) {
            if (words_count.count(w) == 0) words_count[w] = 1;
            else words_count[w] += 1;
        }

        sequences.push_back(words_final);
    }


    vector<vector<int>> getSequencesIds(){
        return sequences_ids;
    }

    void padding(vector<int> &ids, int max_size){

        int padding_nums = max_size - ids.size();
        if (padding_nums > 0){
            for (int i=0; i<padding_nums; i++){
                ids.push_back(PAD_ID);
            }
        }
    }

    void paddingAll(int max_size){
        for (int i=0; i<sequences_ids.size(); i++){

            this->padding(sequences_ids[i], max_size);
        }
    }

    void toOneHot(int v_size, float *data, int id, int col, bool ignore){

        for(int i=0; i<v_size; i++){
            if (i==id && !ignore) data[IDX2F(i, col, v_size)] = 1.;
            else data[IDX2F(i, col, v_size)] = 0.;
        }
    }

    vector<vector<string>> getSequences(){
        return sequences;
    }

    string toWord(int id){
        return idmap_reverse[id];
    }
    int toId(string w){
        return idmap[w];
    }

};



#endif /* WORD_EMBED_H_ */
