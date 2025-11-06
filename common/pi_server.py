import dataclasses
import logging
import socket
import tyro
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    env: str
    port: int = 8000
    record: bool = False


class PolicyMetadata:
    def __init__(self, config: str, checkpoint: str):
        self.config = config
        self.checkpoint = checkpoint


POLICY_DICT = {
    "base_clothes": PolicyMetadata(
        config="base_clothes",
        checkpoint="checkpoints/base_clothes/0804/20000"
    ),
    "base_clothes_pick": PolicyMetadata(
        config="base_clothes_pick",
        checkpoint="checkpoints/base_clothes_pick/0808/59999"
    ),
    "base_clothes_pro": PolicyMetadata(
        config="base_clothes_pro",
        checkpoint="checkpoints/base_clothes_pro/0813/49999"
    ),
    "pi05": PolicyMetadata(
        config="pi05_realman_transfer",
        checkpoint="checkpoints/pi05_realman_transfer/0910/5999"
    ),
}


def create_policy(args: Args) -> _policy.Policy:
    return _policy_config.create_trained_policy(
        _config.get_config(POLICY_DICT[args.env].config), POLICY_DICT[args.env].checkpoint
    )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args: Args = tyro.cli(Args)
    main(args)
