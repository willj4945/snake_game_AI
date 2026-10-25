import pickle

import pytest
import torch

from model import Linear_QNet, QTrainer


@pytest.fixture
def net():
    torch.manual_seed(0)
    return Linear_QNet(11, 256, 3)


def params(model):
    return [p.detach().clone() for p in model.parameters()]


def changed(before, model):
    return any(not torch.equal(b, a) for b, a in zip(before, model.parameters()))


def test_forward_shapes(net):
    assert net(torch.zeros(11)).shape == (3,)
    assert net(torch.zeros(5, 11)).shape == (5, 3)


def test_load_without_checkpoint_returns_defaults(net, in_tmp_dir):
    assert net.load() == (0, 0)


def test_save_and_load_round_trip(net, in_tmp_dir):
    net.save(record=12, n_games=34)
    assert (in_tmp_dir / 'model' / 'model.pth').exists()

    restored = Linear_QNet(11, 256, 3)
    assert restored.load() == (12, 34)
    for a, b in zip(net.parameters(), restored.parameters()):
        assert torch.equal(a, b)


def test_load_legacy_state_dict(net, in_tmp_dir):
    (in_tmp_dir / 'model').mkdir()
    torch.save(net.state_dict(), in_tmp_dir / 'model' / 'model.pth')

    restored = Linear_QNet(11, 256, 3)
    assert restored.load() == (0, 0)
    for a, b in zip(net.parameters(), restored.parameters()):
        assert torch.equal(a, b)


class _NotAllowed:
    pass


def test_load_refuses_arbitrary_pickled_objects(net, in_tmp_dir):
    # weights_only=True stops a tampered checkpoint from running code on load.
    (in_tmp_dir / 'model').mkdir()
    torch.save({'model_state': net.state_dict(), 'record': 0, 'n_games': 0, 'x': _NotAllowed()},
               in_tmp_dir / 'model' / 'model.pth')
    with pytest.raises(pickle.UnpicklingError):
        Linear_QNet(11, 256, 3).load()


def test_train_step_single_transition_updates_weights(net):
    trainer = QTrainer(net, lr=0.01, gamma=0.9)
    before = params(net)
    trainer.train_step([0] * 11, [1, 0, 0], 10, [1] * 11, False)
    assert changed(before, net)


def test_train_step_batch_updates_weights(net):
    trainer = QTrainer(net, lr=0.01, gamma=0.9)
    before = params(net)
    states = ([0] * 11, [1] * 11)
    actions = ([1, 0, 0], [0, 0, 1])
    trainer.train_step(states, actions, (10, -10), states[::-1], (False, True))
    assert changed(before, net)


def test_train_step_moves_prediction_toward_reward(net):
    trainer = QTrainer(net, lr=0.01, gamma=0.9)
    state = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    for _ in range(200):
        trainer.train_step(state, [0, 1, 0], 10, state, True)  # terminal: target is the reward
    assert net(torch.tensor(state, dtype=torch.float))[1].item() == pytest.approx(10, abs=0.5)
