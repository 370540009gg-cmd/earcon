# -*- coding: utf-8 -*-
"""Toy environment: a treasure maze with hidden traps.

A self-contained, zero-dependency environment for the discrete-action
JitRL path. Rules: reach the key, pick it up, unlock the chest. Traps
teleport you back to the start - their positions are only knowable
through experience, which is precisely what memory should learn.

Rewards are dense on purpose (trap = -0.3 immediately): our experiments
showed that sparse, terminal-only rewards leave the advantage signal too
weak for shallow judges to learn from.
"""

DIRECTIONS = {
    "north": (-1, 0),
    "south": (1, 0),
    "east": (0, 1),
    "west": (0, -1),
}
SPECIAL_ACTIONS = ["pick key", "take treasure", "smash jar"]
ALL_ACTIONS = list(DIRECTIONS.keys()) + SPECIAL_ACTIONS

MAPS = {
    # optimal: south, south, pick, east, east, take (6 steps)
    "map1": {"rows": 2, "cols": 3, "start": (0, 0), "key": (1, 0),
             "treasure": (1, 2), "jar": (0, 2)},
    # mirrored
    "map2": {"rows": 2, "cols": 3, "start": (0, 2), "key": (1, 2),
             "treasure": (1, 0), "jar": (0, 0)},
    # traps on the intuitive routes; only experience can avoid them
    "trap1": {"rows": 3, "cols": 3, "start": (0, 0), "key": (2, 0),
              "treasure": (2, 2), "jar": (0, 2),
              "traps": [(0, 1), (1, 1)], "optimal": 6},
    "trap2": {"rows": 3, "cols": 3, "start": (0, 2), "key": (2, 2),
              "treasure": (2, 0), "jar": (0, 0),
              "traps": [(0, 1), (1, 1)], "optimal": 6},
}


def optimal_steps(map_id):
    cfg = MAPS[map_id]
    if "optimal" in cfg:
        return cfg["optimal"]
    d = (abs(cfg["start"][0] - cfg["key"][0]) + abs(cfg["start"][1] - cfg["key"][1])
         + abs(cfg["key"][0] - cfg["treasure"][0])
         + abs(cfg["key"][1] - cfg["treasure"][1]))
    return d + 2


class TreasureMaze:
    def __init__(self, map_id, max_steps=30):
        cfg = MAPS[map_id]
        self.map_id = map_id
        self.rows, self.cols = cfg["rows"], cfg["cols"]
        self.start = cfg["start"]
        self.key_pos, self.treasure_pos, self.jar_pos = (
            cfg["key"], cfg["treasure"], cfg["jar"])
        self.traps = set(cfg.get("traps", []))
        self.max_steps = max_steps
        self.trap_falls = 0
        self.last_event = None
        self.reset()

    def reset(self):
        self.pos = self.start
        self.has_key = False
        self.key_taken = False
        self.steps = 0
        self.done = False
        self.success = False
        self.trap_falls = 0
        self.last_event = None
        return self.observe()

    def observe(self, event=None):
        r, c = self.pos
        parts = ["You are in a %dx%d maze, at row %d col %d "
                 "(top-left is 0,0)." % (self.rows, self.cols, r, c)]
        if self.pos == self.key_pos and not self.key_taken:
            parts.append("There is a key on the ground.")
        if self.pos == self.treasure_pos:
            parts.append("A locked chest stands here."
                         if not self.has_key else
                         "A chest stands here; you have the key.")
        if self.pos == self.jar_pos:
            parts.append("A jar sits in the corner.")
        parts.append("Bag: %s." % ("key" if self.has_key else "empty"))
        if event:
            parts.append(event)
        return " ".join(parts)

    def abstract_state(self):
        """Card state: composed cell token rXcY avoids the Jaccard pitfall
        where (0,1) and (1,0) tokenize to identical sets."""
        r, c = self.pos
        return "map:%s|cell:r%dc%d|bag:%s" % (
            self.map_id, r, c, "key" if self.has_key else "empty")

    def step(self, action):
        """Returns (observation, done, success, step_reward)."""
        assert not self.done
        self.steps += 1
        event, reward = None, 0.0

        if action in DIRECTIONS:
            dr, dc = DIRECTIONS[action]
            nr, nc = self.pos[0] + dr, self.pos[1] + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                self.pos = (nr, nc)
                event = "You go %s." % action
                if self.pos in self.traps:
                    self.trap_falls += 1
                    self.pos = self.start
                    reward = -0.3
                    event = ("You go %s - the floor gives way, a trap! "
                             "You are back at the start." % action)
            else:
                event = "You go %s but hit a wall." % action
        elif action == "pick key":
            if self.pos == self.key_pos and not self.key_taken:
                self.has_key = True
                self.key_taken = True
                reward = 0.5
                event = "You pick up the key."
            else:
                event = "You grope around; nothing here."
        elif action == "take treasure":
            if self.pos == self.treasure_pos:
                if self.has_key:
                    self.done = True
                    self.success = True
                    reward = 1.0
                    event = "You unlock the chest and take the treasure!"
                else:
                    event = "The chest is locked."
            else:
                event = "No chest here."
        elif action == "smash jar":
            event = ("The jar shatters; nothing inside."
                     if self.pos == self.jar_pos else "No jar here.")
        else:
            event = "Invalid action."

        if not self.done and self.steps >= self.max_steps:
            self.done = True
            self.success = False
            reward = -1.0
            event = (event or "") + " Out of steps; failed."

        self.last_event = event
        return self.observe(event), self.done, self.success, reward
