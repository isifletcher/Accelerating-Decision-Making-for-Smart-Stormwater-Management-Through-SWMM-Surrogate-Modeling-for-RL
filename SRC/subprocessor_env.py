import multiprocessing as mp
from src.connor_creek_env import ConnorCreekEnv

def env_worker(pipe_conn, env_config):
    try:
        env = ConnorCreekEnv(**env_config)
        
        init_data = {
            "observation_space_shape": env.observation_space.shape,
            "action_space_nvec": env.action_space.nvec,
            "T": env.T
        }
        pipe_conn.send(("init_data", init_data))
        
        while True:
            command, data = pipe_conn.recv()
            if command == "reset":
                pipe_conn.send(env.reset())
            elif command == "step":
                pipe_conn.send(env.step(data))
            elif command == "close":
                env.close()
                pipe_conn.close()
                break
            else:
                raise NotImplementedError
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        if 'env' in locals():
            env.close()

class SubprocessEnvWrapper:
    def __init__(self, env_config):
        self.parent_conn, self.child_conn = mp.Pipe()
        self.process = mp.Process(target=env_worker, args=(self.child_conn, env_config))
        self.process.start()
        
        command, init_data = self.parent_conn.recv()
        if command != "init_data":
            raise RuntimeError("Subprocess env failed to initialize correctly.")
            
        self.observation_space = type('', (), {'shape': init_data["observation_space_shape"]})()
        self.action_space = type('', (), {'nvec': init_data["action_space_nvec"]})()
        
        self.T = init_data["T"]
        self.mode = env_config['mode']

    def reset(self):
        self.parent_conn.send(("reset", None))
        return self.parent_conn.recv()

    def step(self, action):
        self.parent_conn.send(("step", action))
        return self.parent_conn.recv()

    def close(self):
        try:
            self.parent_conn.send(("close", None))
            self.parent_conn.close()
            self.process.join(timeout=5)
        except (EOFError, BrokenPipeError):
            pass
        finally:
            if self.process.is_alive():
                self.process.terminate()
