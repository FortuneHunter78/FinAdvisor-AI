import { FC, PropsWithChildren } from 'react';
import { Button, buttonProps } from './button';
import { Link } from 'react-router';

const LinkButton: FC<{ to: string } & buttonProps & PropsWithChildren> = ({
  to,
  children,
  ...props
}) => (
  <Link to={to}>
    <Button {...props}>{children}</Button>
  </Link>
);

export default LinkButton;
