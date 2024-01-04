# kd
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0 -b 0.9 --trial 1 
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0 -b 0.9 --trial 2 
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0 -b 0.9 --trial 3 

# kd1proj
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd1proj --model_s resnet8x4 -r 0.1 -a 0 -b 0.9 --trial 1 
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd1proj --model_s resnet8x4 -r 0.1 -a 0 -b 0.9 --trial 2 
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd1proj --model_s resnet8x4 -r 0.1 -a 0 -b 0.9 --trial 3 

# ours
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ours --model_s resnet8x4 -a 0 -b 25 --trial 1  
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ours --model_s resnet8x4 -a 0 -b 25 --trial 2  
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ours --model_s resnet8x4 -a 0 -b 25 --trial 3  


