import random
from constants import TOKENS

def generate_trajectory(task_type, max_steps=10):
    """
    Generates an expert trajectory for a 5x5 GridWorld.
    task_type: "treasure" or "escape"
    Returns a sequence of tokens.
    Format: <TASK> <R_START> <Sx> <Sy> <Tx/Cx> <Ty/Cy> <A> <R> <Sx> ...
    """
    grid_size = 5
    
    # Random starting positions
    ax, ay = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
    ox, oy = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
    
    # Ensure they don't start at the exact same spot
    while ax == ox and ay == oy:
        ox, oy = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
        
    trajectory = []
    
    if task_type == "treasure":
        trajectory.append(TOKENS["<TASK_TREASURE>"])
    else:
        trajectory.append(TOKENS["<TASK_ESCAPE>"])
        
    # Initial reward is neutral
    trajectory.append(TOKENS["<R_NEU>"])
    
    for step in range(max_steps):
        # Add state: Agent X, Agent Y, Other X, Other Y
        trajectory.extend([TOKENS[str(ax)], TOKENS[str(ay)], TOKENS[str(ox)], TOKENS[str(oy)]])
        
        # Check termination condition BEFORE action
        if task_type == "treasure" and ax == ox and ay == oy:
            break
        if task_type == "escape" and ax == ox and ay == oy:
            break
            
        # Determine optimal action
        actions = []
        # Up (dy=-1), Down (dy=1), Left (dx=-1), Right (dx=1)
        if task_type == "treasure":
            if ax < ox: actions.append(("<RIGHT>", 1, 0))
            elif ax > ox: actions.append(("<LEFT>", -1, 0))
            if ay < oy: actions.append(("<DOWN>", 0, 1))
            elif ay > oy: actions.append(("<UP>", 0, -1))
        else: # Escape
            best_dist = -1
            for act_name, dx, dy in [("<UP>", 0, -1), ("<DOWN>", 0, 1), ("<LEFT>", -1, 0), ("<RIGHT>", 1, 0), ("<STAY>", 0, 0)]:
                nx, ny = ax + dx, ay + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    dist = abs(nx - ox) + abs(ny - oy)
                    if dist > best_dist:
                        best_dist = dist
                        actions = [(act_name, dx, dy)]
                    elif dist == best_dist:
                        actions.append((act_name, dx, dy))
                        
        if not actions:
            actions = [("<STAY>", 0, 0)]
            
        # Pick one optimal action
        act_name, dx, dy = random.choice(actions)
        trajectory.append(TOKENS[act_name])
        
        # Update agent state
        ax += dx
        ay += dy
        
        # Update chaser state if escape
        if task_type == "escape":
            c_actions = []
            if ox < ax: c_actions.append((1, 0))
            elif ox > ax: c_actions.append((-1, 0))
            if oy < ay: c_actions.append((0, 1))
            elif oy > ay: c_actions.append((0, -1))
            if c_actions:
                cdx, cdy = random.choice(c_actions)
                ox += cdx
                oy += cdy
                
        # Determine reward
        if task_type == "treasure":
            if ax == ox and ay == oy:
                reward = "<R_POS>"
            else:
                reward = "<R_NEU>"
        else:
            if ax == ox and ay == oy:
                reward = "<R_NEG>"
            elif step == max_steps - 1:
                reward = "<R_POS>"
            else:
                reward = "<R_NEU>"
                
        trajectory.append(TOKENS[reward])
        
        if reward in ["<R_POS>", "<R_NEG>"]:
            # Terminal state reached, add final state and break
            trajectory.extend([TOKENS[str(ax)], TOKENS[str(ay)], TOKENS[str(ox)], TOKENS[str(oy)]])
            break
            
    return trajectory

def make_dt_batch(batch_size, max_steps, device):
    import torch
    from constants import PAD
    
    pairs = []
    tasks = []
    for _ in range(batch_size):
        task_type = random.choice(["treasure", "escape"])
        tasks.append(task_type)
        traj = generate_trajectory(task_type, max_steps=max_steps)
        # For language modeling, x is the sequence up to the last token, y is shifted by 1
        x_seq = traj[:-1]
        y_seq = traj[1:]
        pairs.append((x_seq, y_seq))
        
    max_len = max(len(x) for x, _ in pairs)
    x = torch.full((batch_size, max_len), PAD, dtype=torch.long)
    y = torch.full((batch_size, max_len), -100, dtype=torch.long) # -100 for ignore_index
    
    for i, (xi, yi) in enumerate(pairs):
        x[i, :len(xi)] = torch.tensor(xi)
        # Only compute loss on Action tokens?
        # Actually, standard Decision Transformer predicts actions, but training as a standard LM on everything is fine and simpler.
        # But predicting state is not controlled by the agent (it's environment dynamics).
        # We can just train to predict everything (like Trajectory Transformer) or mask out states/rewards.
        # Let's train to predict everything for simplicity, but maybe we only care about action prediction accuracy.
        y[i, :len(yi)] = torch.tensor(yi)
        
    return x.to(device), y.to(device), tasks
