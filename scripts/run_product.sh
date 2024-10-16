#python main.py --dataset "computers" --manifold "sphere" "sphere" --use_product --embed_dim 16 16 --lr_cls 0.005 --w_decay_cls 0.0001
#
#python main.py --dataset "computers" --manifold "lorentz" "sphere" --use_product --embed_dim 16 16 --lr_cls 0.005 --w_decay_cls 0.0001
#
#python main.py --dataset "computers" --manifold "lorentz" "lorentz" --use_product --embed_dim 16 16 --lr_cls 0.005 --w_decay_cls 0.0001
#
#python main.py --dataset "photo" --manifold "sphere" "sphere" --use_product --embed_dim 16 16 --lr_cls 0.005 --w_decay_cls 0.0001
#
#python main.py --dataset "photo" --manifold "lorentz" "sphere" --use_product --embed_dim 16 16 --lr_cls 0.005 --w_decay_cls 0.0001
#
#python main.py --dataset "photo" --manifold "lorentz" "lorentz" --use_product --embed_dim 16 16 --lr_cls 0.005 --w_decay_cls 0.0001

python main.py --task "NC" --dataset "CS" --manifold "sphere" "sphere" --use_product --embed_dim 16 16 --lr_cls 0.005 --w_decay_cls 0.0001

python main.py --task "NC" --dataset "CS" --manifold "lorentz" "sphere" --use_product --embed_dim 16 16 --lr_cls 0.005 --w_decay_cls 0.0001

python main.py --task "NC" --dataset "CS" --manifold "lorentz" "lorentz" --use_product --embed_dim 16 16 --lr_cls 0.005 --w_decay_cls 0.0001

python main.py --task "NC" --dataset "Physics" --manifold "sphere" "sphere" --use_product --embed_dim 16 16 --lr_cls 0.005 --w_decay_cls 0.0001

python main.py --task "NC" --dataset "Physics" --manifold "lorentz" "sphere" --use_product --embed_dim 16 16 --lr_cls 0.005 --w_decay_cls 0.0001

python main.py --task "NC" --dataset "Physics" --manifold "lorentz" "lorentz" --use_product --embed_dim 16 16 --lr_cls 0.005 --w_decay_cls 0.0001