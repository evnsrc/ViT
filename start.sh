#! /bin/bash

git pull


# train1
python init_train.py
cp weights.pth $HOME/public/train1/weights1.pth
python train_again.py
cp weights.pth $HOME/public/train1/weights2.pth
python train_again.py
cp weights.pth $HOME/public/train1/weights3.pth
python train_again.py
cp weights.pth $HOME/public/train1/weights4.pth
python train_again.py
cp weights.pth $HOME/public/train1/weights5.pth
python train_again.py
cp weights.pth $HOME/public/train1/weights6.pth
python train_again.py
cp weights.pth $HOME/public/train1/weights7.pth
python train_again.py
cp weights.pth $HOME/public/train1/weights8.pth
python train_again.py
cp weights.pth $HOME/public/train1/weights9.pth
python train_again.py
cp weights.pth $HOME/public/train1/weights10.pth


# train 2
python init_train2.py
cp weights.pth $HOME/public/train2/weights1.pth
python train_again2.py
cp weights.pth $HOME/public/train2/weights2.pth
python train_again2.py
cp weights.pth $HOME/public/train2/weights3.pth
python train_again2.py
cp weights.pth $HOME/public/train2/weights4.pth
python train_again2.py
cp weights.pth $HOME/public/train2/weights5.pth
python train_again2.py
cp weights.pth $HOME/public/train2/weights6.pth
python train_again2.py
cp weights.pth $HOME/public/train2/weights7.pth
python train_again2.py
cp weights.pth $HOME/public/train2/weights8.pth
python train_again2.py
cp weights.pth $HOME/public/train2/weights9.pth
python train_again2.py
cp weights.pth $HOME/public/train2/weights10.pth


# train 3
python init_train3.py
cp weights.pth $HOME/public/train3/weights1.pth
python train_again3.py
cp weights.pth $HOME/public/train3/weights2.pth
python train_again3.py
cp weights.pth $HOME/public/train3/weights3.pth
python train_again3.py
cp weights.pth $HOME/public/train3/weights4.pth
python train_again3.py
cp weights.pth $HOME/public/train3/weights5.pth
python train_again3.py
cp weights.pth $HOME/public/train3/weights6.pth
python train_again3.py
cp weights.pth $HOME/public/train3/weights7.pth
python train_again3.py
cp weights.pth $HOME/public/train3/weights8.pth
python train_again3.py
cp weights.pth $HOME/public/train3/weights9.pth
python train_again3.py
cp weights.pth $HOME/public/train3/weights10.pth

