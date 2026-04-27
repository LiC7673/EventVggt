from finetune_event_global_token_common import make_hydra_main


run = make_hydra_main(event_token_weight=0.0, global_token_weight=0.1)


if __name__ == "__main__":
    run()
