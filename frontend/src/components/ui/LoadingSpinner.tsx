interface LoadingSpinnerProps {
  className?: string;
  sizeClassName?: string;
}

export default function LoadingSpinner({
  className = 'flex items-center justify-center h-64',
  sizeClassName = 'w-8 h-8',
}: LoadingSpinnerProps) {
  return (
    <div className={className}>
      <div className={`${sizeClassName} border-2 border-flame border-t-transparent rounded-full animate-spin`} />
    </div>
  );
}
