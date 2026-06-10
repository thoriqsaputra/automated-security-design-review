import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import { ChevronRight } from 'lucide-react';

const routeLabels: Record<string, string> = {
  designs: 'Designs',
  standards: 'Standards',
  reviews: 'Reviews',
};

export default function TopBar() {
  const { pathname } = useLocation();
  const segments = pathname.split('/').filter(Boolean);

  const crumbs = segments.map((seg, i) => ({
    label: routeLabels[seg] || (isNaN(Number(seg)) ? seg.replace(/_/g, ' ') : `#${seg}`),
    path: '/' + segments.slice(0, i + 1).join('/'),
  }));

  return (
    <header className="sticky top-0 z-30 h-14 bg-midnight/80 backdrop-blur-xl border-b border-surface-border flex items-center px-6 pl-20 gap-6 justify-between">
      <div className="flex items-center gap-6">
        <nav className="flex items-center gap-1 text-sm">
          <Link to="/" className="text-text-muted hover:text-text-primary transition-colors">
            Home
          </Link>
          {crumbs.map((c, i) => (
            <React.Fragment key={c.path}>
              <ChevronRight size={14} className="text-text-muted" />
              {i === crumbs.length - 1 ? (
                <span className="text-text-primary font-medium capitalize">{c.label}</span>
              ) : (
                <Link to={c.path} className="text-text-muted hover:text-text-primary transition-colors capitalize">
                  {c.label}
                </Link>
              )}
            </React.Fragment>
          ))}
        </nav>
      </div>
    </header>
  );
}
