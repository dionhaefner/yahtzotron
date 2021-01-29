from functools import partial

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import rlax


@partial(jax.jit, static_argnums=(0,))
def loss(network, weights, observations):
    """"Actor-critic loss."""
    logits, values = network.apply(observations)
    td_errors = rlax.td_lambda(
        v_tm1=values[:-1],
        r_t=trajectory.rewards,
        discount_t=trajectory.discounts * discount,
        v_t=values[1:],
        lambda_=jnp.array(td_lambda),
    )
    critic_loss = jnp.mean(td_errors**2)
    actor_loss = rlax.policy_gradient_loss(
        logits_t=logits[:-1],
        a_t=trajectory.actions,
        adv_t=td_errors,
        w_t=jnp.ones_like(td_errors))

    return actor_loss + critic_loss


@partial(jax.jit, static_argnums=(0,))
def sgd_step(state: TrainingState,
            trajectory: sequence.Trajectory):
      """Does a step of SGD over a trajectory."""
      gradients = jax.grad(loss_fn)(state.params, trajectory)
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)
      return TrainingState(params=new_params, opt_state=new_opt_state)

    # Initialize network parameters and optimiser state.
    init, forward = hk.without_apply_rng(hk.transform(network, apply_rng=True))
    dummy_observation = jnp.zeros((1, *obs_spec.shape), dtype=jnp.float32)
    initial_params = init(next(rng), dummy_observation)
    initial_opt_state = optimizer.init(initial_params)

    # Internalize state.
    self._state = TrainingState(initial_params, initial_opt_state)
    self._forward = jax.jit(forward)
    self._buffer = sequence.Buffer(obs_spec, action_spec, sequence_length)
    self._sgd_step = sgd_step
    self._rng = rng

  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
    """Selects actions according to a softmax policy."""
    key = next(self._rng)
    observation = timestep.observation[None, ...]
    logits, _ = self._forward(self._state.params, observation)
    action = jax.random.categorical(key, logits).squeeze()
    return int(action)

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: base.Action,
      new_timestep: dm_env.TimeStep,
  ):
    """Adds a transition to the trajectory buffer and periodically does SGD."""
    self._buffer.append(timestep, action, new_timestep)
    if self._buffer.full() or new_timestep.last():
      trajectory = self._buffer.drain()
      self._state = self._sgd_step(self._state, trajectory)


def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray,
                  seed: int = 0) -> base.Agent:
  """Creates an actor-critic agent with default hyperparameters."""

  def network(inputs: jnp.ndarray) -> Tuple[Logits, Value]:
    flat_inputs = hk.Flatten()(inputs)
    torso = hk.nets.MLP([64, 64])
    policy_head = hk.Linear(action_spec.num_values)
    value_head = hk.Linear(1)
    embedding = torso(flat_inputs)
    logits = policy_head(embedding)
    value = value_head(embedding)
    return logits, jnp.squeeze(value, axis=-1)

  return ActorCritic(
      obs_spec=obs_spec,
      action_spec=action_spec,
      network=network,
      optimizer=optax.adam(3e-3),
      rng=hk.PRNGSequence(seed),
      sequence_length=32,
      discount=0.99,
      td_lambda=0.9,
  )