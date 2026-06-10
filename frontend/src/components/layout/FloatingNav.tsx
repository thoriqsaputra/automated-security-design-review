import { NavLink } from 'react-router-dom';
import { FileText, ShieldCheck } from 'lucide-react';

const links = [
  { to: '/designs', label: 'Designs', icon: FileText },
  { to: '/standards', label: 'Standards', icon: ShieldCheck },
];

export default function FloatingNav() {
  return (
    <nav className="fixed left-4 top-1/2 -translate-y-1/2 z-50 flex flex-col gap-3">
      {links.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          className={({ isActive }) =>
            `group flex items-center h-12 rounded-full shadow-lg transition-all duration-300 overflow-hidden ${
              isActive
                ? 'bg-gradient-to-r from-crimson to-flame text-white'
                : 'bg-surface-card text-text-secondary hover:text-text-primary hover:bg-surface-hover border border-surface-border'
            } w-12 hover:w-36`
          }
        >
          <div className="flex items-center justify-center min-w-[3rem] h-12">
            <Icon size={20} className="shrink-0" />
          </div>
          <span className="text-sm font-medium whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 delay-100">
            {label}
          </span>
        </NavLink>
      ))}
    </nav>
  );
}
