import { useEffect, useMemo, useReducer, useRef, useState } from 'react';
import {
  getReviewDebateStreamUrl,
  type DebateSnapshotPayload,
  type DebateStreamState,
  type DebateUpdatePayload,
} from '../../../api/reviews';

interface DebateStreamStateShape {
  connected: boolean;
  reviewStatus: string | null;
  errorMessage: string | null;
  lastEventId: string | null;
  debatesById: Record<string, DebateStreamState>;
}

type Action =
  | { type: 'snapshot'; payload: DebateSnapshotPayload }
  | { type: 'updates'; payloads: DebateUpdatePayload[] }
  | { type: 'connected'; value: boolean };

const initialState: DebateStreamStateShape = {
  connected: false,
  reviewStatus: null,
  errorMessage: null,
  lastEventId: null,
  debatesById: {},
};

function sortDebates(left: DebateStreamState, right: DebateStreamState) {
  const rank = (status: string) => {
    switch (status) {
      case 'running':
        return 0;
      case 'pending':
        return 1;
      case 'completed':
        return 2;
      case 'failed':
        return 3;
      case 'cancelled':
        return 4;
      default:
        return 5;
    }
  };
  const rankDiff = rank(left.status) - rank(right.status);
  if (rankDiff !== 0) {
    return rankDiff;
  }
  const leftTime = left.updated_at ? Date.parse(left.updated_at) : 0;
  const rightTime = right.updated_at ? Date.parse(right.updated_at) : 0;
  return rightTime - leftTime;
}

function reducer(state: DebateStreamStateShape, action: Action): DebateStreamStateShape {
  switch (action.type) {
    case 'connected':
      return { ...state, connected: action.value };
    case 'snapshot': {
      const debatesById = Object.fromEntries(
        action.payload.debates.map((debate) => [debate.debate_id, debate]),
      );
      return {
        ...state,
        reviewStatus: action.payload.review_status,
        errorMessage: action.payload.error_message || null,
        lastEventId: action.payload.last_event_id,
        debatesById,
      };
    }
    case 'updates': {
      let nextState = state;
      let debatesById = state.debatesById;
      let reviewStatus = state.reviewStatus;
      let errorMessage = state.errorMessage;

      for (const payload of action.payloads) {
        if (payload.review_status !== undefined) {
          reviewStatus = payload.review_status || null;
        }
        if (payload.error_message !== undefined) {
          errorMessage = payload.error_message || null;
        }
        if (payload.debate) {
          if (debatesById === state.debatesById) {
            debatesById = { ...state.debatesById };
          }
          debatesById[payload.debate.debate_id] = payload.debate;
        }
        if (payload.debates?.length) {
          if (debatesById === state.debatesById) {
            debatesById = { ...state.debatesById };
          }
          for (const debate of payload.debates) {
            debatesById[debate.debate_id] = debate;
          }
        }
      }

      if (
        debatesById === state.debatesById
        && reviewStatus === state.reviewStatus
        && errorMessage === state.errorMessage
      ) {
        return state;
      }
      nextState = {
        ...state,
        debatesById,
        reviewStatus,
        errorMessage,
      };
      return nextState;
    }
    default:
      return state;
  }
}

export function useReviewDebateStream(
  reviewId: number | null,
  reviewStatus: string | null,
) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const queueRef = useRef<DebateUpdatePayload[]>([]);
  const flushTimerRef = useRef<number | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);

  useEffect(() => {
    if (!reviewId) {
      dispatch({ type: 'snapshot', payload: { review_id: reviewId || 0, review_status: reviewStatus, updated_at: null, last_event_id: null, debates: [] } });
      return undefined;
    }

    const source = new EventSource(getReviewDebateStreamUrl(reviewId));
    const flushQueue = () => {
      flushTimerRef.current = null;
      if (!queueRef.current.length) {
        return;
      }
      const payloads = queueRef.current.splice(0, queueRef.current.length);
      dispatch({ type: 'updates', payloads });
    };
    const enqueue = (payload: DebateUpdatePayload) => {
      queueRef.current.push(payload);
      if (flushTimerRef.current !== null) {
        return;
      }
      flushTimerRef.current = window.setTimeout(flushQueue, 120);
    };

    source.addEventListener('open', () => {
      dispatch({ type: 'connected', value: true });
      setStreamError(null);
    });
    source.addEventListener('error', () => {
      dispatch({ type: 'connected', value: false });
      setStreamError('Live debate stream disconnected.');
    });
    source.addEventListener('snapshot', (event) => {
      const payload = JSON.parse((event as MessageEvent).data) as DebateSnapshotPayload;
      dispatch({ type: 'snapshot', payload });
    });
    source.addEventListener('debate.updated', (event) => {
      enqueue(JSON.parse((event as MessageEvent).data) as DebateUpdatePayload);
    });
    source.addEventListener('debates.seeded', (event) => {
      enqueue(JSON.parse((event as MessageEvent).data) as DebateUpdatePayload);
    });
    source.addEventListener('review.status', (event) => {
      enqueue(JSON.parse((event as MessageEvent).data) as DebateUpdatePayload);
    });

    return () => {
      source.close();
      dispatch({ type: 'connected', value: false });
      if (flushTimerRef.current !== null) {
        window.clearTimeout(flushTimerRef.current);
      }
      queueRef.current = [];
    };
  }, [reviewId, reviewStatus]);

  const debates = useMemo(
    () => Object.values(state.debatesById).sort(sortDebates),
    [state.debatesById],
  );

  return {
    connected: state.connected,
    streamError,
    reviewStatus: state.reviewStatus || reviewStatus,
    errorMessage: state.errorMessage,
    debates,
  };
}
