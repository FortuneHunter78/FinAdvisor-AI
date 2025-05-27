import type { FC, ReactNode } from 'react';
import { HoverCard, HoverCardContent, HoverCardTrigger } from './hover-card';
import { Button } from './button';
import { Info } from 'lucide-react';

const HoverInfoCard: FC<{ info: ReactNode; icon?: ReactNode }> = ({
  info,
  icon,
}) => (
  <HoverCard>
    <HoverCardTrigger asChild>
      <Button variant='ghost' size='icon'>
        {icon || <Info size={20} />}
      </Button>
    </HoverCardTrigger>
    <HoverCardContent
      className='w-80 text-xs border border-sidebar-border rounded-2xl bg-popover'
      side='bottom'
    >
      {info}
    </HoverCardContent>
  </HoverCard>
);

export default HoverInfoCard;
