from ray.rllib.algorithms.callbacks import DefaultCallbacks

class MyCustomCallbacks(DefaultCallbacks):
    """
    This custom callback considers a single environnement evaluation during the
    episode step. If needs to be corrected when multiple workers are used.
    Otherwise all the workers are appended to a a single rendered result.
    """

    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        episode.user_data["custom_metrics"] = []

        if worker.config["in_evaluation"]:
            if not hasattr(self, 'eval_num'):
                self.eval_num = 0
            else:
                self.eval_num += 1

    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):

        if worker.config["in_evaluation"]:
            #print(episode.total_agent_steps)
            env = base_env.get_sub_environments()[0]
            env.render_custom(self.eval_num)

        # Record some custom metrics
        #custom_metric = episode.last_info_for()["some_metric"]
        #episode.user_data["custom_metrics"].append(custom_metric)

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        pass
        # if worker.config["in_evaluation"]:
        #     pdb.set_trace()

        #pass
        #print("Episode ending")
        #custom_metrics = episode.user_data["custom_metrics"]
        #episode.custom_metrics["mean_custom_metric"] = np.mean(custom_metrics)
        #logger.info(f"Episode ended with custom metrics: {episode.custom_metrics}")

    def on_postprocess_trajectory(self, **kwargs):
        pass
        # for step in trajectory.timesteps:
        #     if step.step_type == "training":
        #         iteration = step.step_id
        #         print(f"Current iteration: {iteration}")
    
    def on_train_result(self,**kwargs):
        #pdb.set_trace()
        self.iter_n = kwargs['algorithm'].iteration
        # iteration = trainer.iteration if hasattr(trainer, "iteration") else 0
        # print(f"Training iteration {iteration} completed with result: {result}")