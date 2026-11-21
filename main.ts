// sokoban-solver-01 entry: a push-optimal Sokoban solver built on room decomposition.
//
// A level from this competition's generator is a one-square-wide hallway with small rooms
// (3 or 4 boxes each) hanging off it, one door per room. A box pushed out of its room can
// never come back, so the rooms are independent push problems and the level's push optimum
// is the sum of the rooms' optima. This solver therefore:
//
//   1. cuts the dead squares of the player's row (the hallway) out of the floor; each
//      connected part that holds boxes is one room;
//   2. solves every room on its own with a breadth-first search over pushes (uniform cost,
//      so the first solution found is push-optimal), pruning boxes on dead squares and
//      frozen 2x2 blocks, the other rooms' boxes standing as walls;
//   3. joins the rooms' move strings, the player walking on from where the last room
//      left it.
//
// Fallbacks, in order, if a step cannot finish inside its work budget: an A* search over
// pushes for that room (same pruning, guided by summed single-box push distances), then
// an A* over pushes for the whole level, then an empty answer. Every budget is a count
// of states, never the clock, and every tie is broken by square order, so two runs of
// this program on the same set write byte-identical answers.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Work budgets, in states kept or expanded. Counts, never time. */
const ROOM_BFS_STATES = 2_000_000;
const ROOM_ASTAR_EXPANSIONS = 2_000_000;
const BOARD_ASTAR_EXPANSIONS = 1_500_000;

/** The four directions in a fixed order: [dx, dy, walk letter, push letter]. */
const DIRS: ReadonlyArray<readonly [number, number, string, string]> = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** A level from XSB text. Short rows and the outermost frame count as walls. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * The live squares: those from which a box alone can be pushed onto some goal, found by
 * pulling boxes back from the goals over open floor. A box on any other square is lost.
 */
function liveSquares(w: number, wall: Uint8Array, goals: number[]): Uint8Array {
  const live = new Uint8Array(wall.length);
  const queue: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Scratch for reachability floods: a stamp grid and a stack, shared by every search. */
class Reach {
  seen: Int32Array;
  stack: Int32Array;
  stamp = 0;
  constructor(size: number) {
    this.seen = new Int32Array(size);
    this.stack = new Int32Array(size);
  }
  /** Marks the region the player reaches from `from` around walls and boxes; returns its lowest square. */
  flood(w: number, wall: Uint8Array, occupied: Uint8Array, from: number): number {
    const stamp = ++this.stamp;
    const seen = this.seen;
    const stack = this.stack;
    let n = 0;
    let low = from;
    stack[n++] = from;
    seen[from] = stamp;
    while (n > 0) {
      const p = stack[--n];
      if (p < low) low = p;
      const up = p - w;
      const dn = p + w;
      const lf = p - 1;
      const rt = p + 1;
      if (wall[up] === 0 && occupied[up] === 0 && seen[up] !== stamp) { seen[up] = stamp; stack[n++] = up; }
      if (wall[dn] === 0 && occupied[dn] === 0 && seen[dn] !== stamp) { seen[dn] = stamp; stack[n++] = dn; }
      if (wall[lf] === 0 && occupied[lf] === 0 && seen[lf] !== stamp) { seen[lf] = stamp; stack[n++] = lf; }
      if (wall[rt] === 0 && occupied[rt] === 0 && seen[rt] !== stamp) { seen[rt] = stamp; stack[n++] = rt; }
    }
    return low;
  }
  /** Whether square `s` carries the current flood's stamp. */
  has(s: number): boolean {
    return this.seen[s] === this.stamp;
  }
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(w: number, wall: Uint8Array, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters. */
function walkTo(w: number, wall: Uint8Array, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: target out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/** The square the player stands on after playing `moves` from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) {
    const c = ch.toLowerCase();
    p += c === 'u' ? -w : c === 'd' ? w : c === 'l' ? -1 : 1;
  }
  return p;
}

interface Solution {
  pushes: number;
  moves: string;
}

/**
 * Rebuilds the move string of a search from its state chain: `count` states, state s
 * keeping its sorted boxes at `boxesArena[s * n ..]`, the square the player stands on
 * after the push that made it at `playerAfter[s]`, and its parent at `parent[s]`.
 * The walk to each push is the shortest one around the parent state's boxes.
 */
function reconstruct(
  w: number,
  wall: Uint8Array,
  n: number,
  boxesArena: Int32Array,
  playerAfter: Int32Array,
  parent: Int32Array,
  startPlayer: number,
  startBoxes: ArrayLike<number>,
  found: number,
): Solution {
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]) path.push(s);
  path.reverse();
  const occupied = new Uint8Array(wall.length);
  let moves = '';
  let player = startPlayer;
  let prevBoxes: ArrayLike<number> = startBoxes;
  for (const s of path) {
    const base = s * n;
    let from = -1;
    let to = -1;
    for (let i = 0; i < n; i++) {
      const b = boxesArena[base + i];
      let inPrev = false;
      for (let j = 0; j < n; j++) if (prevBoxes[j] === b) { inPrev = true; break; }
      if (!inPrev) to = b;
    }
    for (let j = 0; j < n; j++) {
      const b = prevBoxes[j];
      let inNext = false;
      for (let i = 0; i < n; i++) if (boxesArena[base + i] === b) { inNext = true; break; }
      if (!inNext) from = b;
    }
    const d = to - from;
    let letter = '';
    for (const [dx, dy, , pushLetter] of DIRS) {
      if (dy * w + dx === d) { letter = pushLetter; break; }
    }
    for (let j = 0; j < n; j++) occupied[prevBoxes[j]] = 1;
    moves += walkTo(w, wall, occupied, player, from - d);
    for (let j = 0; j < n; j++) occupied[prevBoxes[j]] = 0;
    moves += letter;
    player = from;
    prevBoxes = boxesArena.subarray(base, base + n);
  }
  return { pushes: path.length, moves };
}

/**
 * The push-optimal solution of one (sub-)level by breadth-first search over pushes, or
 * null when the search keeps `maxStates` states without solving it. A state is the
 * boxes' squares plus the player's reachable region, named by its lowest square; the
 * states come out of the queue in order of pushes made, so the first solved one is
 * optimal. Boxes pushed onto dead squares and frozen 2x2 blocks are pruned, which no
 * solvable state ever needs.
 */
function solvePushBFS(level: Level, maxStates: number): Solution | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const n = level.boxes.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(w, wall, goals);
  const solved = (arr: Int32Array, base: number): boolean => {
    for (let i = 0; i < n; i++) if (!goal[arr[base + i]]) return false;
    return true;
  };
  if (level.boxes.some((b) => !live[b])) return null;

  // Two floods: `reachHere` holds the expanded state's region while `reachNext` names
  // each child's, so checking one never overwrites the other.
  const reachHere = new Reach(size);
  const reachNext = new Reach(size);
  const occupied = new Uint8Array(size);
  let cap = 1024;
  let boxesArena = new Int32Array(cap * n);
  let playerAfter = new Int32Array(cap);
  let parent = new Int32Array(cap);
  const grow = (): void => {
    cap *= 2;
    const nb = new Int32Array(cap * n);
    nb.set(boxesArena);
    boxesArena = nb;
    const np = new Int32Array(cap);
    np.set(playerAfter);
    playerAfter = np;
    const nq = new Int32Array(cap);
    nq.set(parent);
    parent = nq;
  };
  const index = new Map<number, number>();
  // A state's key: the region's lowest square, then the boxes, as digits in base `size`.
  const keyOf = (low: number, arr: Int32Array, base: number): number => {
    let k = low;
    for (let i = 0; i < n; i++) k = k * size + arr[base + i];
    return k;
  };

  for (let i = 0; i < n; i++) boxesArena[i] = level.boxes[i];
  playerAfter[0] = level.player;
  parent[0] = -1;
  let count = 1;
  for (const b of level.boxes) occupied[b] = 1;
  index.set(keyOf(reachHere.flood(w, wall, occupied, level.player), boxesArena, 0), 0);
  for (const b of level.boxes) occupied[b] = 0;
  if (solved(boxesArena, 0)) return { pushes: 0, moves: '' };

  const child = new Int32Array(n);
  for (let head = 0; head < count; head++) {
    const base = head * n;
    for (let i = 0; i < n; i++) occupied[boxesArena[base + i]] = 1;
    reachHere.flood(w, wall, occupied, playerAfter[head]);
    for (let bi = 0; bi < n; bi++) {
      const b = boxesArena[base + bi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (!reachHere.has(b - d) || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(w, wall, occupied, goal, t);
        let key = -1;
        if (!frozen) {
          // The child's boxes: state boxes with b replaced by t, kept sorted.
          let m = 0;
          for (let i = 0; i < n; i++) {
            const v = boxesArena[base + i];
            if (v !== b) child[m++] = v;
          }
          let ins = m - 1;
          while (ins >= 0 && child[ins] > t) {
            child[ins + 1] = child[ins];
            ins--;
          }
          child[ins + 1] = t;
          key = keyOf(reachNext.flood(w, wall, occupied, b), child, 0);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || index.has(key)) continue;
        if (count >= maxStates) {
          for (let i = 0; i < n; i++) occupied[boxesArena[base + i]] = 0;
          return null;
        }
        if (count >= cap) grow();
        const cb = count * n;
        for (let i = 0; i < n; i++) boxesArena[cb + i] = child[i];
        playerAfter[count] = b;
        parent[count] = head;
        index.set(key, count);
        count++;
        if (solved(boxesArena, cb)) {
          return reconstruct(w, wall, n, boxesArena, playerAfter, parent, level.player, level.boxes, count - 1);
        }
      }
    }
    for (let i = 0; i < n; i++) occupied[boxesArena[base + i]] = 0;
  }
  return null;
}

/**
 * Per-goal push distances: dist[g][s] is the fewest pushes that move a box from square s
 * onto goal g, ignoring the other boxes and the player's reach (found by pulling back
 * from g). Infinity where no push sequence reaches. Used as the A* estimate's table.
 */
function pushDistances(w: number, wall: Uint8Array, goals: number[]): Int32Array {
  const size = wall.length;
  const table = new Int32Array(size * goals.length).fill(-1);
  goals.forEach((g, gi) => {
    const dist = table.subarray(gi * size, (gi + 1) * size);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const x = queue[qi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const y = x - d;
        if (!wall[y] && !wall[y - d] && dist[y] === -1) {
          dist[y] = dist[x] + 1;
          queue.push(y);
        }
      }
    }
  });
  return table;
}

/**
 * A push-optimal solution by A* over pushes, for boards the breadth-first search cannot
 * hold: the estimate is the sum over boxes of each box's own fewest pushes to its nearest
 * goal, which no solution can beat (a push moves one box one step), and it never drops
 * by more than one per push, so the first solved state A* settles is optimal. Same
 * dead-square and frozen-block pruning as the breadth-first search. Null past the
 * expansion budget.
 */
function solvePushAStar(level: Level, maxExpansions: number): Solution | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const n = level.boxes.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(w, wall, goals);
  if (level.boxes.some((b) => !live[b])) return null;
  const dist = pushDistances(w, wall, goals);
  const estimate = (arr: Int32Array, base: number): number => {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const b = arr[base + i];
      let best = -1;
      for (let gi = 0; gi < goals.length; gi++) {
        const v = dist[gi * size + b];
        if (v !== -1 && (best === -1 || v < best)) best = v;
      }
      if (best === -1) return -1;
      sum += best;
    }
    return sum;
  };
  const solved = (arr: Int32Array, base: number): boolean => {
    for (let i = 0; i < n; i++) if (!goal[arr[base + i]]) return false;
    return true;
  };

  // Two floods, as in the breadth-first search: the settled state's region and each
  // child's are named separately so one never overwrites the other.
  const reachHere = new Reach(size);
  const reachNext = new Reach(size);
  const occupied = new Uint8Array(size);
  let cap = 1024;
  let boxesArena = new Int32Array(cap * n);
  let playerAfter = new Int32Array(cap);
  let parent = new Int32Array(cap);
  let gScore = new Int32Array(cap);
  const grow = (): void => {
    cap *= 2;
    const nb = new Int32Array(cap * n);
    nb.set(boxesArena);
    boxesArena = nb;
    const np = new Int32Array(cap);
    np.set(playerAfter);
    playerAfter = np;
    const nq = new Int32Array(cap);
    nq.set(parent);
    parent = nq;
    const ng = new Int32Array(cap);
    ng.set(gScore);
    gScore = ng;
  };
  const keyOf = (low: number, arr: Int32Array, base: number): string => {
    let k = String(low);
    for (let i = 0; i < n; i++) k += ',' + arr[base + i];
    return k;
  };
  const best = new Map<string, number>();

  // The open list: a binary heap of [f, -g, state], ties to the deeper state, then to
  // the earlier one made, so the order is fixed.
  const heap: Array<readonly [number, number, number]> = [];
  const less = (a: readonly [number, number, number], b: readonly [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: readonly [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      const tmp = heap[i];
      heap[i] = heap[p];
      heap[p] = tmp;
      i = p;
    }
  };
  const heapPop = (): readonly [number, number, number] => {
    const top = heap[0];
    const last = heap.pop() as readonly [number, number, number];
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        const tmp = heap[i];
        heap[i] = heap[c];
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };

  for (let i = 0; i < n; i++) boxesArena[i] = level.boxes[i];
  playerAfter[0] = level.player;
  parent[0] = -1;
  gScore[0] = 0;
  let count = 1;
  const h0 = estimate(boxesArena, 0);
  if (h0 === -1) return null;
  for (const b of level.boxes) occupied[b] = 1;
  best.set(keyOf(reachHere.flood(w, wall, occupied, level.player), boxesArena, 0), 0);
  for (const b of level.boxes) occupied[b] = 0;
  if (solved(boxesArena, 0)) return { pushes: 0, moves: '' };
  heapPush([h0, 0, 0]);

  const child = new Int32Array(n);
  let expanded = 0;
  while (heap.length > 0) {
    const [, , s] = heapPop();
    const base = s * n;
    const g = gScore[s];
    for (let i = 0; i < n; i++) occupied[boxesArena[base + i]] = 1;
    const low = reachHere.flood(w, wall, occupied, playerAfter[s]);
    if (best.get(keyOf(low, boxesArena, base)) !== g) {
      for (let i = 0; i < n; i++) occupied[boxesArena[base + i]] = 0;
      continue;
    }
    if (solved(boxesArena, base)) {
      return reconstruct(w, wall, n, boxesArena, playerAfter, parent, level.player, level.boxes, s);
    }
    if (++expanded > maxExpansions) return null;
    for (let bi = 0; bi < n; bi++) {
      const b = boxesArena[base + bi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (!reachHere.has(b - d) || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(w, wall, occupied, goal, t);
        let key = '';
        let h = -1;
        if (!frozen) {
          let m = 0;
          for (let i = 0; i < n; i++) {
            const v = boxesArena[base + i];
            if (v !== b) child[m++] = v;
          }
          let ins = m - 1;
          while (ins >= 0 && child[ins] > t) {
            child[ins + 1] = child[ins];
            ins--;
          }
          child[ins + 1] = t;
          h = estimate(child, 0);
          if (h !== -1) key = keyOf(reachNext.flood(w, wall, occupied, b), child, 0);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || h === -1) continue;
        const seen = best.get(key);
        if (seen !== undefined && seen <= g + 1) continue;
        if (count >= cap) grow();
        const cb = count * n;
        for (let i = 0; i < n; i++) boxesArena[cb + i] = child[i];
        playerAfter[count] = b;
        parent[count] = s;
        gScore[count] = g + 1;
        best.set(key, g + 1);
        heapPush([g + 1 + h, -(g + 1), count]);
        count++;
      }
    }
    for (let i = 0; i < n; i++) occupied[boxesArena[base + i]] = 0;
  }
  return null;
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  done?: boolean;
}

/**
 * The level's rooms: the dead squares of the player's row (the hallway: a box there
 * reaches no goal) are cut out of the floor, and each part left that holds a box is a
 * room. Null when the level is no such set of rooms: a box in the hallway, a goal in a
 * cut square, or a part whose boxes and goals differ in number.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(w, wall, goals);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some((g) => part[g] < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** One room's push-optimal solution: breadth-first first, A* when the breadth-first search outgrows its budget. */
function solveRoom(sub: Level): Solution | null {
  return solvePushBFS(sub, ROOM_BFS_STATES) ?? solvePushAStar(sub, ROOM_ASTAR_EXPANSIONS);
}

/**
 * The level solved room by room: each room's push-optimal solution in turn, the other
 * rooms' boxes standing as walls (on their goals once their room is done), the player
 * walking on from where the last room left it. Falls back to a whole-level A* when the
 * level is no set of rooms or a room defeats both of its searches.
 */
function solveLevel(level: Level): string {
  const rooms = roomsOf(level);
  if (rooms !== null) {
    let player = level.player;
    let pushes = 0;
    let moves = '';
    let ok = true;
    for (const room of rooms) {
      const wall = level.wall.slice();
      for (const other of rooms) {
        if (other === room) continue;
        for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
      }
      const sub: Level = {
        w: level.w,
        h: level.h,
        wall,
        goals: room.goals.slice().sort((a, b) => a - b),
        boxes: room.boxes.slice().sort((a, b) => a - b),
        player,
      };
      const solvedRoom = solveRoom(sub);
      if (solvedRoom === null) {
        ok = false;
        break;
      }
      pushes += solvedRoom.pushes;
      moves += solvedRoom.moves;
      player = endOf(level.w, player, solvedRoom.moves);
      room.done = true;
    }
    if (ok) return moves;
  }
  const whole = solvePushAStar(level, BOARD_ASTAR_EXPANSIONS);
  return whole === null ? '' : whole.moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(parseLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8')));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
