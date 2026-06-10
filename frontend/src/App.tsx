import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import AppShell from './components/layout/AppShell';
import DesignsList from './pages/DesignsList';
import DesignDetail from './pages/DesignDetail';
import StandardsHub from './pages/StandardsHub';
import CategoryDetail from './pages/CategoryDetail';
import ReviewDetail from './pages/ReviewDetail';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/" element={<Navigate to="/designs" replace />} />
          <Route path="/designs" element={<DesignsList />} />
          <Route path="/designs/:id" element={<DesignDetail />} />
          <Route path="/standards" element={<StandardsHub />} />
          <Route path="/standards/:code" element={<CategoryDetail />} />
          <Route path="/designs/:designId/reviews" element={<Navigate to=".." relative="path" replace />} />
          <Route path="/designs/:designId/reviews/:id" element={<ReviewDetail />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
