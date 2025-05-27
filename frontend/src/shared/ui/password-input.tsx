import { FC, useState } from 'react';
import { Input, inputProps } from './input';
import { cn } from '../lib/utils';
import { Button } from './button';
import { EyeIcon, EyeOffIcon } from 'lucide-react';

const ShowPasswordIcon: FC<{ isShowPassword: boolean; size?: number }> = ({
  isShowPassword,
  size = 16,
}) => {
  if (isShowPassword) {
    return <EyeIcon size={size} aria-hidden='true' />;
  }

  return <EyeOffIcon size={size} aria-hidden='true' />;
};

const PasswordInput: FC<inputProps> = ({ className, ...props }) => {
  const [isShowPassword, setIsShowPassword] = useState<boolean>(false);

  return (
    <div className={cn('relative', className)}>
      <Input {...props} type={isShowPassword ? 'text' : 'password'} />
      <Button
        type='button'
        size='icon'
        variant='ghost'
        className='absolute right-0 top-0 h-full'
        onClick={() => setIsShowPassword((prev) => !prev)}
      >
        <ShowPasswordIcon isShowPassword={isShowPassword} />
      </Button>
    </div>
  );
};

export default PasswordInput;
