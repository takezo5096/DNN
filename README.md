# DNN

## 事前作業
* CUDAのインストール
* MeCabのインストール

## コンパイル & 実行
cuMatディレクトリが同列にあることを前提
```bash
cp test.cpp.[処理の種類]　test.cpp
make
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../cuMat
./test
```
#### [処理の種類]
* autoencoder: オートエンコーダー
* cnn: 畳み込みニューラルネットワーク
* iris: IRISデータセットによる層状パーセプトロン
* lstm.sin:LSTMによるsin波再現
* mlp:MNISTによる層状パーセプトロン
* number:LSTMによる数当て
* seq2seq:LSTMによる翻訳モデル

## MNISTのダウンロード(test.cpp.mlp, autoencoder用)
http://yann.lecun.com/exdb/mnist/

train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 

train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 

t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 

t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

の４つのファイルをダウンロード後、DNNディレクトリに解凍

## Cifar-10のダウンロード(test.cpp.cnn用)
https://www.cs.toronto.edu/~kriz/cifar.html

CIFAR-10 binary version (suitable for C programs)

をダウンロード後、DNNディレクトリに解凍。cifar-10-batches-binディレクトリの下にデータが格納されている。

## Tanaka Corpusのダウンロード(test.cpp.seq2seq用)
http://www.edrdg.org/wiki/index.php/Tanaka_Corpus

complete version (UTF-8)をダウンロード後、DNNディレクトリに解凍。examples.utfができる。

norm_tanaka_corpus.pyで必要な部分だけ取り出し、tanaka_corpus_e.txtとtanaka_corpus_j.txtを作成。

sample_tanaka_corpus.pyで作成したデータからランダムに10000件を抜き出して、別ファイルに保存(tanaka_corpus_e_10000.txt, 

tanaka_corpus_j_10000.txt)。これをリネームしておく(tanaka_corpus_e_10000.txt.train, tanaka_corpus_j_10000.txt.train)

同様に、もう一度sample_tanaka_corpus.pyを使って、テスト用のデータを作る(tanaka_corpus_e_10000.txt.test, tanaka_corpus_j_10000.txt.test)

