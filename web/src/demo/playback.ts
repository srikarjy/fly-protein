/** Play/pause logic. It never advances on a bare timer: the next step is requested only after the coordinator reports the current
 *  step settled (its staged animation, if any, finished) and a short dwell has passed, so playback can't outrun what is on screen. */
import type { ReplayStore } from "./store";

export interface Scheduler {
  setTimeout(fn: () => void, ms: number): number;
  clearTimeout(id: number): void;
}

export class Playback {
  playing = false;
  private timer: number | undefined;

  constructor(private store: ReplayStore, private onChange: () => void, private sched: Scheduler = window) {}

  play() {
    if (this.playing) return;
    if (this.store.k >= this.store.engine.budget) this.store.reset();
    this.playing = true;
    this.onChange();
    this.advance();
  }

  pause() {
    this.sched.clearTimeout(this.timer!);
    this.timer = undefined;
    if (!this.playing) return;
    this.playing = false;
    this.onChange();
  }

  toggle() { this.playing ? this.pause() : this.play(); }

  private advance() {
    this.timer = undefined;
    if (!this.playing) return;
    if (!this.store.next("play")) this.pause();
  }

  /** the coordinator calls this when the current step has fully settled; `dwellMs` is how long to hold it before the next step */
  settled(dwellMs: number) {
    if (!this.playing) return;
    if (this.store.k >= this.store.engine.budget) { this.pause(); return; }
    this.sched.clearTimeout(this.timer!);
    this.timer = this.sched.setTimeout(() => this.advance(), dwellMs);
  }
}
