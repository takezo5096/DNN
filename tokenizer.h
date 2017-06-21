/*
 * tokenizer.h
 *
 */

#ifndef TOKENIZER_H_
#define TOKENIZER_H_

#include <mecab.h>
#include <string>

std::vector<std::string> split(const std::string &str, char sep)
{
    std::vector<std::string> v;        // 分割結果を格納するベクター
    auto first = str.begin();              // テキストの最初を指すイテレータ
    while( first != str.end() ) {         // テキストが残っている間ループ
        auto last = first;                      // 分割文字列末尾へのイテレータ
        while( last != str.end() && *last != sep )       // 末尾 or セパレータ文字まで進める
            ++last;
        v.push_back(std::string(first, last));       // 分割文字を出力
        if( last != str.end() )
            ++last;
        first = last;          // 次の処理のためにイテレータを設定
    }
    return v;
}


class Tokenizer {
private:
    MeCab::Tagger *tagger;

public:
    Tokenizer(){
        //tagger = MeCab::createTagger("-Owakati");
        //tagger = MeCab::createTagger("-xunknown -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd");
        tagger = MeCab::createTagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd");
    }
    ~Tokenizer(){
        delete tagger;
    }

    std::vector<std::string> parse(std::string input){


        std::vector<std::string> result;

        const MeCab::Node* node = tagger->parseToNode(input.c_str());

        char buf[1024];

        for (; node; node = node->next) {
            string feature(node->feature);
            //if (feature.find("名詞")==0 || feature.find("未知語")==0){
                strcpy(buf,node->surface);
                buf[node->length]='\0';
                string surface(buf);
                result.push_back(surface);
            //}
        }
        return result;
    }
};




#endif /* TOKENIZER_H_ */
