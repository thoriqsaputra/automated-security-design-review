import { Navigate, useParams } from 'react-router-dom';

export default function ReviewDetail() {
  const { designId, id } = useParams<{ designId: string; id: string }>();
  if (!designId) {
    return <Navigate to="/designs" replace />;
  }
  return <Navigate to={`/designs/${designId}${id ? `?reviewId=${id}` : ''}`} replace />;
}
