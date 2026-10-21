"""
Synthetic multi-domain pretraining data for the GRAM task.

                        >>> DO NOT MODIFY THIS FILE <<<
        (check.py imports it; the checker's baselines depend on it)

Three "domains" share one vocabulary of VOCAB content tokens plus a BOS token:

    0 = core   general-purpose data (80% of training sequences)
    1 = A      dual-use domain A    (10%)
    2 = B      dual-use domain B    (10%)

Every sequence is a first-order Markov chain drawn from a low-entropy
transition table. The core domain is a mixture of N_CORE_TOPICS different
tables ("topics", like the 44 core topics in the paper's SimpleStories
experiment); A and B each have a single table of their own. All tables are
independent, so a domain's statistics are "knowledge" that can only be learned
from that domain's data: a model that has never seen domain A gets a high loss
on A sequences, and a model that has seen it gets a low loss.

A model has to work out which table a sequence comes from by looking at the
context (there is no domain or topic token in the input).

Batches are MIXED: every sequence in a batch has its own domain label. Think of
this as a packed pretraining batch where you cannot choose to make batches
domain-homogeneous.
"""
import torch

VOCAB = 32            # content tokens 0 .. VOCAB-1
BOS = VOCAB           # beginning-of-sequence token
VOCAB_SIZE = VOCAB + 1
SEQ_LEN = 32          # content tokens per sequence; inputs are [BOS] + SEQ_LEN tokens
N_DOMAINS = 3
N_CORE_TOPICS = 4     # the core domain is a mixture of this many tables
DOMAIN_NAMES = ["core", "A", "B"]
DOMAIN_PROBS = torch.tensor([0.8, 0.1, 0.1])
DUAL_USE = [1, 2]     # domain ids that get an auxiliary module

# Training budget shared by every model the checker trains (do not change).
TRAIN_STEPS = 1500
BATCH_SIZE = 64

_DATA_SEED = 1234
_ROW_SCALE = 2.0      # larger -> lower-entropy transition rows
_UNIFORM_MIX = 0.02   # keeps cross-entropies between domains finite


def _make_tables() -> torch.Tensor:
    """(N_CORE_TOPICS + 2, VOCAB, VOCAB) row-stochastic transition tables.
    Tables 0..N_CORE_TOPICS-1 are the core topics, then A, then B."""
    g = torch.Generator().manual_seed(_DATA_SEED)
    logits = torch.randn(N_CORE_TOPICS + 2, VOCAB, VOCAB, generator=g) * _ROW_SCALE
    rows = torch.softmax(logits, dim=-1)
    return (1 - _UNIFORM_MIX) * rows + _UNIFORM_MIX / VOCAB


TABLES = _make_tables()
A_TABLE, B_TABLE = N_CORE_TOPICS, N_CORE_TOPICS + 1


def _sample_chain(table: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    """n sequences of SEQ_LEN content tokens from one transition table."""
    out = torch.empty(n, SEQ_LEN, dtype=torch.long)
    cur = torch.randint(0, VOCAB, (n,), generator=gen)
    out[:, 0] = cur
    for t in range(1, SEQ_LEN):
        cur = torch.multinomial(table[cur], 1, generator=gen).squeeze(1)
        out[:, t] = cur
    return out


def sample_sequences(domains: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """domains: (n,) long. Returns (n, SEQ_LEN + 1) tokens starting with BOS."""
    n = domains.numel()
    toks = torch.empty(n, SEQ_LEN + 1, dtype=torch.long)
    toks[:, 0] = BOS
    # which transition table each sequence uses
    table = torch.randint(0, N_CORE_TOPICS, (n,), generator=gen)   # core: a random topic
    table[domains == 1] = A_TABLE
    table[domains == 2] = B_TABLE
    for k in range(N_CORE_TOPICS + 2):
        idx = (table == k).nonzero(as_tuple=True)[0]
        if idx.numel():
            toks[idx, 1:] = _sample_chain(TABLES[k], idx.numel(), gen)
    return toks


def sample_batch(batch_size: int, gen: torch.Generator):
    """A mixed training batch.

    Returns
        tokens : (batch_size, SEQ_LEN + 1) long, tokens[:, 0] == BOS
        labels : (batch_size,) long, domain id of each sequence
    """
    labels = torch.multinomial(DOMAIN_PROBS, batch_size, replacement=True, generator=gen)
    return sample_sequences(labels, gen), labels


def eval_set(domain: int, n: int = 512, seed: int = 99) -> torch.Tensor:
    """Fixed held-out sequences from one domain, (n, SEQ_LEN + 1)."""
    gen = torch.Generator().manual_seed(seed * 1000 + domain)
    return sample_sequences(torch.full((n,), domain, dtype=torch.long), gen)


def sequence_losses(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Per-sequence mean next-token cross-entropy.

    logits : (B, SEQ_LEN + 1, VOCAB_SIZE) -- logits[:, t] predicts tokens[:, t + 1]
    tokens : (B, SEQ_LEN + 1)
    returns (B,) float
    """
    pred = logits[:, :-1]
    tgt = tokens[:, 1:]
    nll = torch.nn.functional.cross_entropy(
        pred.reshape(-1, pred.size(-1)), tgt.reshape(-1), reduction="none"
    )
    return nll.view(tgt.shape).mean(dim=1)


@torch.no_grad()
def domain_metrics(model, domain: int, active_modules=None, n: int = 512):
    """(mean loss, top-1 next-token accuracy) of `model` on held-out sequences
    from `domain`, with the given auxiliary modules switched on."""
    was_training = model.training
    model.eval()
    toks = eval_set(domain, n)
    logits = model(toks, active_modules=active_modules)
    model.train(was_training)
    loss = sequence_losses(logits, toks).mean().item()
    acc = (logits[:, :-1].argmax(-1) == toks[:, 1:]).float().mean().item()
    return loss, acc


def domain_loss(model, domain: int, active_modules=None, n: int = 512) -> float:
    return domain_metrics(model, domain, active_modules, n)[0]


if __name__ == "__main__":
    # Entropy of each domain's chain vs. cross-entropy of predicting it with
    # another domain's table: the "knowledge gap" the task is about.
    names = [f"core{k}" for k in range(N_CORE_TOPICS)] + ["A", "B"]
    for k, p in enumerate(TABLES):
        ent = -(p * p.log()).sum(-1).mean().item()
        xes = [-(p * q.log()).sum(-1).mean().item() for q in TABLES]
        print(f"table {names[k]:>5}: entropy {ent:.2f} nats, top-1 acc {p.max(-1).values.mean():.2f}; "
              f"cross-entropy under each table: {[round(x, 2) for x in xes]}")
    gen = torch.Generator().manual_seed(0)
    toks, labels = sample_batch(8, gen)
    print("example batch labels:", labels.tolist())
    print("example sequence:", toks[0].tolist())
