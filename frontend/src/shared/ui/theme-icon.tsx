import type { FC } from 'react';
import { themeVariant } from '../lib/theme-provider';
import { MoonStar, Sun } from 'lucide-react';

const ThemeIcon: FC<{ size?: string | number; theme: themeVariant }> = ({
  size = 20,
  theme,
}) => {
  if (theme === 'light') {
    return <MoonStar size={size} />;
  }
  return <Sun size={size} />;
};

export default ThemeIcon;
