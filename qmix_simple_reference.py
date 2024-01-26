from easydict import EasyDict

n_agent = 2
n_landmark = n_agent
collector_env_num = 8
evaluator_env_num = 8
main_config = dict(
    exp_name="reference_qmix",
    env=dict(
        env_family="mpe",
        env_id="simple_reference_v3",
        n_agent=n_agent,
        n_landmark=n_landmark,
        max_cycles=25,
        agent_obs_only=False,
        continuous_actions=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=0,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            agent_num=n_agent,
            obs_shape=21,
            global_obs_shape=32,
            action_shape=50,
            hidden_size_list=[128, 128, 64],
            mixer=True,
        ),
        learn=dict(
            update_per_collect=100,
            batch_size=32,
            learning_rate=0.0005,
            target_update_theta=0.001,
            discount_factor=0.99,
            double_q=True,
        ),
        collect=dict(
            n_sample=600,
            unroll_len=16,
            env_num=collector_env_num,
        ),
        eval=dict(
            env_num=evaluator_env_num,
        ),
        other=dict(
            eps=dict(
                type="exp",
                start=1.0,
                end=0.05,
                decay=100000,
            ),
        ),
    ),
)
main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        import_names=["simple_reference_env"],
        type="petting_zoo",
    ),
    env_manager=dict(type="subprocess"),
    policy=dict(type="qmix"),
)
create_config = EasyDict(create_config)

ptz_simple_spread_qmix_config = main_config
ptz_simple_spread_qmix_create_config = create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline

    serial_pipeline((main_config, create_config), seed=0)
