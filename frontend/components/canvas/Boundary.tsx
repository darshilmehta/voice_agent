"use client";

/** Keeps a canvas failure (a chart chunk that won't load, data a chart can't draw) from taking the page down. */

import { Component, type ReactNode } from "react";

interface Props {
  /** What to show instead; it gets a function that tries the children again. */
  fallback: (retry: () => void, error: Error) => ReactNode;
  children: ReactNode;
}

export class CanvasBoundary extends Component<Props, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error) {
    console.warn("canvas:", error);
  }

  render() {
    if (this.state.error) return this.props.fallback(() => this.setState({ error: null }), this.state.error);
    return this.props.children;
  }
}
