# DNN
cuMatディレクトリが同列にあることを前提
cp test.cpp.[処理の種類]　test.cpp
make
export LD_LIBRARY_PATH=../cuMat$LD_LIBRARY_PATH
./test


[処理の種類]
autoencoder: オートエンコーダー
cnn: 畳み込みニューラルネットワーク
iris: IRISデータセットによる層状パーセプトロン
lstm.sin:LSTMによるsin波再現
mlp:層状パーセプトロン
number:LSTMによる数当て
seq2seq:LSTMによる翻訳モデル
