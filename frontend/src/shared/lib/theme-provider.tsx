import type { FC, PropsWithChildren } from 'react';
import { useEffect, useState } from 'react';
import { Button } from '../ui/button';
import ThemeIcon from '../ui/theme-icon';

export type themeVariant = 'light' | 'dark';

export const ThemeProvider: FC<PropsWithChildren> = ({ children }) => {
  const [theme, setTheme] = useState<themeVariant>(
    localStorage.theme || 'light',
  );

  useEffect(() => {
    document.documentElement.classList.toggle(
      'dark',
      localStorage.theme === 'dark' ||
        (!('theme' in localStorage) &&
          window.matchMedia('(prefers-color-scheme: dark)').matches),
    );
  }, [theme]);

  const switchTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';

    localStorage.theme = newTheme;
    setTheme(newTheme);
  };

  return (
    <>
      {children}

      <Button
        className='fixed right-5 bottom-5'
        size='icon'
        onClick={switchTheme}
      >
        <ThemeIcon theme={theme} />
      </Button>
    </>
  );
};
