import React from 'react';

interface Props {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  hover?: boolean;
}

export default function Card({ children, className = '', onClick, hover = false }: Props) {
  return (
    <div
      onClick={onClick}
      className={`
        bg-surface-card border border-surface-border rounded-xl p-5
        transition-all duration-200
        ${hover ? 'hover:border-burgundy hover:bg-surface-hover cursor-pointer hover:shadow-lg hover:shadow-burgundy/10' : ''}
        ${onClick ? 'cursor-pointer' : ''}
        ${className}
      `}
    >
      {children}
    </div>
  );
}
