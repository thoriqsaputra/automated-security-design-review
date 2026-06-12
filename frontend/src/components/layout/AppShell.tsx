import { Outlet } from 'react-router-dom';
import FloatingNav from './FloatingNav';
import TopBar from './TopBar';

export default function AppShell() {
  return (
    <div className="flex min-h-screen relative">
      <FloatingNav />
      <div className="flex-1 flex flex-col w-full">
        <TopBar />
        <main className="flex-1 p-6 animate-fade-in pl-20">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
