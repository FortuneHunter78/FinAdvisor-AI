import { FC } from 'react';
import { Button, buttonProps } from './button';
import { Loader2 } from 'lucide-react';

const LoadingButton: FC<{ isLoading?: boolean } & buttonProps> = ({
  isLoading,
  children,
  ...props
}) => (
  <Button {...props} disabled={isLoading}>
    {isLoading && <Loader2 className='animate-spin' />}
    {children}
  </Button>
);

export default LoadingButton;
