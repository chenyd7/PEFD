# fetch pre-trained teacher models

mkdir -p save/models/

cd save/models

mkdir -p resnet32x4_vanilla
wget http://shape2prog.csail.mit.edu/repo/resnet32x4_vanilla/ckpt_epoch_240.pth
mv ckpt_epoch_240.pth resnet32x4_vanilla/

mkdir -p RESNET34_vanilla
wget https://download.pytorch.org/models/resnet34-333f7ec4.pth
mv resnet34-333f7ec4.pth RESNET34_vanilla/ckpt_epoch_90.pth

cd ../..