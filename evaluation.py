from dizoo.petting_zoo.config.ptz_simple_spread_qmix_config import main_config, create_config
from ding.entry import eval


def main():
    ckpt_path = './ptz_simple_spread_qmix_seed0_240126_220714/ckpt\ckpt_best.pth.tar'
    replay_path = './replay_videos'
    eval((main_config, create_config), seed=0, load_path=ckpt_path, replay_path=replay_path)


if __name__ == "__main__":
    main()