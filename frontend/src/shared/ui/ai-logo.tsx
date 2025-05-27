import type { FC } from 'react';
import { cn } from '../lib/utils';
import { Cpu } from 'lucide-react';

const AILogo: FC<{ className?: string; size?: number }> = ({
  className,
  size = 40,
}) => (
  <div
    className={cn(
      'flex gap-2 items-center justify-center text-[40px]',
      className,
    )}
  >
    <Cpu size={size} />
    <h3 className={`font-jolly-lodger text-center`}>
      FinAdvisor.
      <span className='text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500'>
        AI
      </span>
    </h3>
  </div>
);

export default AILogo;
